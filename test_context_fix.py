"""Tests for context_fix — the four Honcho context fixes' pure logic.

Run with ``python3 -m unittest test_context_fix`` (stdlib only, no pytest and
no Hermes tree needed: ``context_fix`` deliberately imports nothing from the
provider package, so it loads standalone by path).

The middleware rewrites live LLM request payloads, so the cases that matter
most here are the ones where it must NOT act: an in-flight tool loop, an
out-of-band user message, an over-cap payload, a malformed list. Every one of
those must pass the original request through untouched.
"""

from __future__ import annotations

import importlib.util
import os
import types
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "honcho_context_fix_under_test", Path(__file__).with_name("context_fix.py")
)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


def msg(role: str, content: str = "", **extra) -> dict:
    return {"role": role, "content": content, **extra}


def slim(messages: list):
    """Return the kept roles, or None when the middleware passed through."""
    result = cf.on_llm_request(request={"messages": messages, "model": "m"})
    if result is None:
        return None
    return [m["role"] for m in result["request"]["messages"]]


class SlimMessagesTest(unittest.TestCase):
    def test_keeps_system_block_and_current_turn(self):
        self.assertEqual(
            slim([
                msg("system", "S"),
                msg("user", "u1"), msg("assistant", "a1"),
                msg("user", "u2"), msg("assistant", "a2"),
                msg("user", "current"),
            ]),
            ["system", "user"],
        )

    def test_keeps_in_flight_tool_loop_intact(self):
        # Dropping any of these would break the tool-call/result pairing.
        self.assertEqual(
            slim([
                msg("system", "S"),
                msg("user", "u1"), msg("assistant", "a1"),
                msg("user", "current"),
                msg("assistant", "", tool_calls=[{"id": "1"}]),
                msg("tool", "result", tool_call_id="1"),
            ]),
            ["system", "user", "assistant", "tool"],
        )

    def test_out_of_band_user_message_is_not_a_turn_boundary(self):
        # A user message queued after a tool result sits mid-turn; the real
        # turn start (carrying Honcho's injection) is further back.
        self.assertEqual(
            slim([
                msg("system", "S"),
                msg("user", "u1"), msg("assistant", "a1"),
                msg("user", "current"),
                msg("assistant", "", tool_calls=[{"id": "1"}]),
                msg("tool", "result", tool_call_id="1"),
                msg("user", "queued mid-turn"),
            ]),
            ["system", "user", "assistant", "tool", "user"],
        )

    def test_multiple_leading_system_messages_all_kept(self):
        self.assertEqual(
            slim([
                msg("system", "S1"), msg("developer", "S2"),
                msg("user", "u1"), msg("assistant", "a1"),
                msg("user", "current"),
            ]),
            ["system", "developer", "user"],
        )

    def test_no_system_block(self):
        self.assertEqual(
            slim([msg("user", "u1"), msg("assistant", "a1"), msg("user", "current")]),
            ["user"],
        )

    def test_passes_through_when_nothing_to_drop(self):
        self.assertIsNone(slim([msg("system", "S"), msg("user", "u1"), msg("assistant", "a1")]))

    def test_passes_through_short_list(self):
        self.assertIsNone(slim([msg("system", "S"), msg("user", "u1")]))

    def test_passes_through_when_still_over_cap(self):
        # The system block and the current turn are both undroppable, so an
        # over-cap payload is one this middleware cannot fix — fail open.
        self.assertIsNone(slim([
            msg("system", "x" * 200_000),
            msg("user", "u1"), msg("assistant", "a1"), msg("user", "current"),
        ]))

    def test_tolerates_malformed_entries(self):
        self.assertIsNone(cf.on_llm_request(request={"messages": ["junk", None, msg("user", "u")]}))

    def test_non_dict_request_passes_through(self):
        self.assertIsNone(cf.on_llm_request(request=None))

    def test_preserves_other_request_keys(self):
        request = {
            "messages": [
                msg("system", "S"), msg("user", "u1"),
                msg("assistant", "a1"), msg("user", "current"),
            ],
            "model": "claude", "temperature": 0.7,
        }
        result = cf.on_llm_request(request=request)
        self.assertEqual(result["request"]["model"], "claude")
        self.assertEqual(result["request"]["temperature"], 0.7)
        # The original payload must not be mutated in place.
        self.assertEqual(len(request["messages"]), 4)

    def test_kill_switch_disables_middleware(self):
        os.environ[cf.DISABLE_ENV] = "1"
        try:
            self.assertIsNone(slim([
                msg("system", "S"), msg("user", "u1"),
                msg("assistant", "a1"), msg("user", "current"),
            ]))
        finally:
            del os.environ[cf.DISABLE_ENV]
        self.assertTrue(context_fix_enabled_again := not cf.context_fix_disabled())
        self.assertTrue(context_fix_enabled_again)


class FakeMessage:
    def __init__(self, content, peer_id, metadata=None):
        self.content = content
        self.peer_id = peer_id
        self.metadata = metadata
        self.id = "msg-1"
        self.created_at = "2026-01-01T00:00:00Z"


class FakeContext:
    def __init__(self, summary, messages):
        self.summary = summary
        self.messages = messages


class SynthesizeSummaryTest(unittest.TestCase):
    def test_builds_transcript_in_chronological_order(self):
        ctx = cf.synthesize_summary(
            FakeContext(None, [FakeMessage("hi", "u-1"), FakeMessage("hello", "hermes")]),
            {"u-1": "Eva", "hermes": "assistant"},
        )
        content = ctx.summary.content
        self.assertIn("Eva: hi", content)
        self.assertIn("assistant: hello", content)
        self.assertLess(content.index("Eva: hi"), content.index("assistant: hello"))

    def test_never_overrides_a_real_summary(self):
        real = types.SimpleNamespace(content="the real summary")
        ctx = cf.synthesize_summary(FakeContext(real, [FakeMessage("hi", "u-1")]))
        self.assertIs(ctx.summary, real)

    def test_no_messages_leaves_context_unchanged(self):
        self.assertIsNone(cf.synthesize_summary(FakeContext(None, [])).summary)

    def test_empty_message_contents_are_skipped(self):
        self.assertIsNone(cf.synthesize_summary(FakeContext(None, [FakeMessage("", "u-1")])).summary)

    def test_metadata_peer_name_wins_over_label_map(self):
        ctx = cf.synthesize_summary(
            FakeContext(None, [FakeMessage("yo", "u-1", {"peer_name": "Bob"})]), {"u-1": "Eva"}
        )
        self.assertIn("Bob: yo", ctx.summary.content)

    def test_unknown_peer_falls_back_to_peer_id(self):
        ctx = cf.synthesize_summary(FakeContext(None, [FakeMessage("yo", "stranger")]), {"u-1": "Eva"})
        self.assertIn("stranger: yo", ctx.summary.content)

    def test_respects_character_budget_keeping_newest(self):
        messages = [FakeMessage(f"msg{i} " + "x" * 400, "u-1") for i in range(100)]
        ctx = cf.synthesize_summary(FakeContext(None, messages))
        content = ctx.summary.content
        self.assertLess(len(content), cf.synth_max_chars() + 1_000)
        self.assertIn("msg99", content)  # newest kept
        self.assertNotIn("msg0 ", content)  # oldest budgeted out

    def test_long_message_is_truncated(self):
        ctx = cf.synthesize_summary(FakeContext(None, [FakeMessage("y" * 5_000, "u-1")]))
        self.assertIn("…", ctx.summary.content)

    def test_fails_open_on_garbage_context(self):
        self.assertIsNone(cf.synthesize_summary(None))
        self.assertEqual(cf.synthesize_summary("not a context"), "not a context")


class EnvTunableTest(unittest.TestCase):
    def _with_env(self, name, value, fn):
        previous = os.environ.get(name)
        os.environ[name] = value
        try:
            return fn()
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_defaults(self):
        self.assertEqual(cf.max_tokens_cap(), cf.DEFAULT_MAX_TOKENS)
        self.assertEqual(cf.synth_max_chars(), cf.DEFAULT_SYNTH_MAX_CHARS)
        self.assertEqual(cf.sync_join_timeout(), cf.DEFAULT_SYNC_JOIN_TIMEOUT)

    def test_overrides(self):
        self.assertEqual(self._with_env(cf.MAX_TOKENS_ENV, "1234", cf.max_tokens_cap), 1234)
        self.assertEqual(self._with_env(cf.SYNC_JOIN_TIMEOUT_ENV, "0", cf.sync_join_timeout), 0.0)

    def test_invalid_values_fall_back_to_defaults(self):
        self.assertEqual(self._with_env(cf.MAX_TOKENS_ENV, "abc", cf.max_tokens_cap), cf.DEFAULT_MAX_TOKENS)
        self.assertEqual(self._with_env(cf.MAX_TOKENS_ENV, "-5", cf.max_tokens_cap), cf.DEFAULT_MAX_TOKENS)
        self.assertEqual(
            self._with_env(cf.SYNC_JOIN_TIMEOUT_ENV, "nope", cf.sync_join_timeout),
            cf.DEFAULT_SYNC_JOIN_TIMEOUT,
        )

    def test_kill_switch_accepts_common_truthy_spellings(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            self.assertTrue(self._with_env(cf.DISABLE_ENV, value, cf.context_fix_disabled), value)
        for value in ("0", "false", "", "off"):
            self.assertFalse(self._with_env(cf.DISABLE_ENV, value, cf.context_fix_disabled), value)


if __name__ == "__main__":
    unittest.main()
