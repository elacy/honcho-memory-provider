"""Context fixes for the Honcho memory provider.

These were previously applied from outside the provider by the
``honcho-context-fix`` user plugin, which monkey-patched the Honcho SDK and
the provider class at runtime. They now live in the provider itself; this
module holds the parts that are not tied to a single call site:

* the ``HONCHO_CONTEXT_FIX_DISABLE`` kill switch and the env-tunable budgets
  the fixes read,
* :func:`synthesize_summary` — build a session summary out of the recent
  messages the server already returned, for the (common) case where Honcho
  has not generated one yet.  Called from
  ``session.HonchoSessionManager.get_prefetch_context``.
* :func:`on_llm_request` — ``llm_request`` middleware that drops the raw
  conversation history Hermes replays on every API call, and
  :func:`make_llm_request_middleware`, which binds it to a provider so it only
  runs in modes where Honcho actually injects context.  Registered in
  ``register()``.

The two remaining fixes live at their call sites because that is all they
are: ``session.py`` passes ``tokens=`` to ``Session.context`` so the server
builds the context to fit, and ``__init__.py``'s ``sync_turn`` joins its
write thread so the write commits before the next turn's read (for the
write-through ``writeFrequency`` modes — see :func:`sync_join_timeout`).

Everything here is fail-open: on any unexpected shape or error the caller's
data is returned untouched.  A wrong answer from these fixes must never cost
a turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
import types
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Kill switch and tunables. Names are kept from the honcho-context-fix plugin
# so existing deployments keep working after the fixes moved in-tree.
DISABLE_ENV = "HONCHO_CONTEXT_FIX_DISABLE"
MAX_TOKENS_ENV = "HONCHO_CONTEXT_FIX_MAX_TOKENS"
SYNTH_CHARS_ENV = "HONCHO_CONTEXT_FIX_SYNTH_CHARS"
SYNC_JOIN_TIMEOUT_ENV = "HONCHO_CONTEXT_FIX_SYNC_JOIN_TIMEOUT"

DEFAULT_MAX_TOKENS = 40_000
DEFAULT_SYNTH_MAX_CHARS = 6_000  # ~1.7k tokens of recent-turn transcript
DEFAULT_SYNC_JOIN_TIMEOUT = 30.0  # seconds; bounded so a wedged server can't hang the worker

# Conservative chars-per-token ratio for budget enforcement (JSON-ish text
# tokenizes denser than prose). Only used to decide when to trim.
_CHARS_PER_TOKEN = 3.5

# Per-message cap inside a synthesized transcript.
_SYNTH_PER_MESSAGE_CHARS = 500
_SYNTH_SUMMARY_TYPE = "recent_messages_synth"


def context_fix_disabled() -> bool:
    """Return True when the operator has switched these fixes off."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


def max_tokens_cap() -> int:
    """Token ceiling for the payload the llm_request middleware lets through."""
    return _positive_int_env(MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS)


def synth_max_chars() -> int:
    """Character budget for a synthesized session summary."""
    return _positive_int_env(SYNTH_CHARS_ENV, DEFAULT_SYNTH_MAX_CHARS)


def sync_join_timeout() -> float:
    """Bound on how long ``sync_turn`` waits for its write thread to commit.

    Bounded so a wedged Honcho server can't hang the memory manager's single
    worker forever. ``0`` disables the wait (pure fire-and-forget, the
    behavior before this fix).

    Only consulted for the write-through ``writeFrequency`` modes
    (``turn``/``session``/N). Under ``async`` the write is handed to the
    session manager's queue, so there is nothing a join here could wait on;
    see ``__init__.py``'s ``_sync_join_timeout``.
    """
    raw = os.environ.get(SYNC_JOIN_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return DEFAULT_SYNC_JOIN_TIMEOUT


# ---------------------------------------------------------------------------
# Synthesize a session summary when the server has none
# ---------------------------------------------------------------------------


def _speaker_label(message: Any, labels: Optional[dict[str, str]] = None) -> str:
    """Best-effort speaker name for a Honcho Message (or duck-typed stand-in).

    A message that carries its author's name in metadata wins; otherwise the
    caller-supplied peer-id map names the two session peers, so a generated
    peer id (a numeric Telegram user id, ``user-telegram-<chat>``) doesn't end
    up as the speaker name in the transcript.
    """
    meta = getattr(message, "metadata", None)
    if isinstance(meta, dict):
        name = meta.get("peer_name") or meta.get("peerName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    peer_id = str(getattr(message, "peer_id", "") or "")
    if not peer_id:
        return "?"
    if labels:
        return labels.get(peer_id) or peer_id
    return peer_id


def synthesize_summary(ctx: Any, labels: Optional[dict[str, str]] = None) -> Any:
    """Attach a transcript-derived summary to ``ctx`` when the server has none.

    Honcho only generates a session summary when the message count is
    divisible by ``messages_per_short_summary`` (20) or
    ``messages_per_long_summary`` (60). Per-thread sessions — every Telegram
    thread gets its own — almost never reach that, so ``ctx.summary`` is null
    for effectively every turn and the session half of the injected context is
    empty, losing task continuity. The response does carry ``ctx.messages``,
    so build the summary out of those instead of dropping them.

    A real summary is never overridden. Fail-open: on any problem the context
    is returned unchanged.
    """
    try:
        existing = getattr(ctx, "summary", None)
        if existing is not None and getattr(existing, "content", None):
            return ctx  # real summary present — never override

        messages = getattr(ctx, "messages", None) or []
        if not messages:
            return ctx

        max_chars = synth_max_chars()
        lines: list[str] = []
        used = 0
        # Budget most-recent-first (the newest turns matter most), then
        # restore chronological order for the rendered transcript.
        for message in reversed(list(messages)):
            content = getattr(message, "content", None)
            if not content:
                continue
            text = str(content).strip().replace("\r", " ").replace("\n", " ")
            if len(text) > _SYNTH_PER_MESSAGE_CHARS:
                text = text[:_SYNTH_PER_MESSAGE_CHARS] + "…"
            line = f"- {_speaker_label(message, labels)}: {text}"
            if used + len(line) > max_chars and lines:
                break
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return ctx
        lines.reverse()

        body = (
            f"[No Honcho summary generated yet for this session — "
            f"verbatim transcript of the last {len(lines)} message(s) instead]\n"
            + "\n".join(lines)
        )
        synth = _build_summary(body, messages[-1])

        try:
            ctx.summary = synth  # pydantic default: assignment w/o revalidation
        except Exception:
            try:
                ctx = ctx.model_copy(update={"summary": synth})
            except Exception:
                return ctx  # cannot attach — fail open

        logger.debug(
            "Honcho synthesized a session summary from %d recent message(s) (~%d chars)",
            len(lines),
            len(body),
        )
        return ctx
    except Exception as e:
        logger.debug("Honcho summary synthesis failed, passing context through: %s", e)
        return ctx


def _build_summary(body: str, last_message: Any) -> Any:
    """Return a Summary-shaped object carrying ``body``.

    Uses the SDK model when it is importable and accepts these values, so the
    object stays valid for anything that type-checks it; falls back to a
    duck-typed stand-in, which is all the read paths (``summary.content``)
    actually need.
    """
    try:
        from honcho.session_context import Summary

        return Summary(
            content=body,
            message_id=str(getattr(last_message, "id", "") or ""),
            summary_type=_SYNTH_SUMMARY_TYPE,
            created_at=str(getattr(last_message, "created_at", "") or ""),
            token_count=_estimate_tokens(body),
        )
    except Exception:
        return types.SimpleNamespace(
            content=body,
            summary_type=_SYNTH_SUMMARY_TYPE,
            message_id=None,
            created_at=None,
            token_count=None,
        )


# ---------------------------------------------------------------------------
# llm_request middleware: drop replayed history, cap the payload
# ---------------------------------------------------------------------------

# Hermes fences whatever a memory provider returns from ``prefetch()`` in this
# tag (``agent/memory_manager.py::build_memory_context_block``) and appends it
# to the API copy of the current turn's user message
# (``agent/turn_context.py::compose_api_user_content``). Its presence in the
# current turn is what tells this middleware that Honcho's context is on the
# wire and the replayed history is therefore redundant.
MEMORY_CONTEXT_TAG = "<memory-context>"
# Matched the way Hermes matches its own fence (``_FENCE_TAG_RE`` in
# memory_manager.py): case-insensitive and tolerant of whitespace inside the
# tag, so a future change to how the block is rendered can't silently turn
# this detector — and with it the whole middleware — into a no-op.
_MEMORY_CONTEXT_RE = re.compile(r"<\s*memory-context\s*>", re.IGNORECASE)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _message_tokens(msg: dict) -> int:
    total = 4  # per-message overhead
    content = msg.get("content")
    if isinstance(content, str):
        total += _estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or ""
                total += _estimate_tokens(text if isinstance(text, str) else json.dumps(text))
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            total += _estimate_tokens(json.dumps(tool_calls, default=str))
        except Exception:
            total += 50
    return total


def _leading_system_count(messages: list) -> int:
    n = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
            n += 1
        else:
            break
    return n


def _find_turn_start(messages: list, sys_count: int) -> int:
    """Index of the user message that started the current turn.

    Scan backwards from the end of the list. A user message at index j is a
    turn boundary when the message before it is a system/developer message or
    an assistant message (i.e. the previous turn's final reply or the session
    start). Out-of-band user messages queued mid-turn sit after tool results,
    so they are NOT boundaries — keep scanning past them until the real
    turn-start user message (the one carrying this turn's Honcho injection).
    Returns -1 when no boundary is found — the caller then passes through.

    Only ``system``/``developer``/``user``/``assistant``/``tool`` are
    recognized. Anything else means the payload has a shape this middleware
    was not written against, so it gives up on the whole request rather than
    guessing a boundary: an unknown role sitting between an assistant tool
    call and its result would otherwise be taken for the turn start, and the
    slice would ship the orphaned result without its call (API 400).
    """
    i = len(messages) - 1
    while i >= sys_count:
        msg = messages[i]
        role = msg.get("role") if isinstance(msg, dict) else None
        if role in ("assistant", "tool"):
            i -= 1
            continue
        if role in ("system", "developer"):
            # A system/developer message *past* the leading block is a
            # host-injected reminder appended to the tail of the turn in
            # flight, not a boundary. Skipping it is what keeps a trailing
            # reminder from being mistaken for the current turn — which would
            # slice the real user message and its tool loop away. The reminder
            # itself still ships: everything from the turn start onward is
            # kept.
            i -= 1
            continue
        if role == "user":
            prev = messages[i - 1] if i > 0 else None
            prev_role = prev.get("role") if isinstance(prev, dict) else None
            if prev_role in ("system", "developer", "assistant"):
                return i
            # Mid-turn queued user message (after a tool result) — keep
            # scanning backwards for the real turn start.
            i -= 1
            continue
        # Unknown/unexpected role — this payload isn't the shape we reason
        # about. Fail open on the whole request.
        return -1
    return -1


def _message_text(msg: dict) -> str:
    """Flatten a message's content to text (string or multimodal part list).

    Multimodal content is walked too, even though Hermes only ever appends the
    fence to string content (``compose_api_user_content`` returns None for
    anything else): a part list that does carry the block should be honored,
    and one that doesn't reads as "no injection", which is the safe answer.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts)
    return ""


def _turn_carries_injection(messages: list, turn_start: int) -> bool:
    """True when the current turn carries a Honcho ``<memory-context>`` block.

    This is the load-bearing precondition for stripping: the replayed history
    may only be dropped when Honcho's injection is there to replace it. It is
    absent whenever the provider injected nothing this turn — tools-only
    recall, ``injectionFrequency: first-turn`` past turn 1, a trivial prompt,
    a cron/flush context, paused auth, an image-only (multimodal) user message
    that Hermes never fences — and dropping the history in any of those cases
    would leave the model with no conversation at all.

    Scoped to the current turn on purpose. Earlier turns replay their own
    fenced blocks (Hermes' ``api_content`` sidecar keeps the prompt-cache
    prefix byte-stable), so a whole-list search would always match.
    """
    for msg in messages[turn_start:]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            if _MEMORY_CONTEXT_RE.search(_message_text(msg)):
                return True
    return False


def _slim_messages(request: dict, cap: int) -> Optional[dict]:
    """Return a slimmed copy of ``request``, or None when nothing is safe to do."""
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return None  # nothing to strip / too small to reason about
    if not all(isinstance(m, dict) for m in messages):
        return None  # malformed entries — fail open, never ship them to the API

    sys_count = _leading_system_count(messages)
    turn_start = _find_turn_start(messages, sys_count)
    if turn_start <= 0:
        return None  # no user message bounds the tail — pass through

    if not _turn_carries_injection(messages, turn_start):
        logger.debug(
            "Honcho injected no context into this turn; leaving the replayed "
            "history in place"
        )
        return None

    kept = messages[:sys_count] + messages[turn_start:]
    if len(kept) >= len(messages):
        return None  # nothing dropped

    # Enforce the cap. Neither half of what's left is droppable: the current
    # turn (from turn_start onward) carries the in-flight tool loop, and the
    # leading system block defines the agent. So an over-cap payload is one we
    # can't fix here — send the original and let Hermes' own compaction run.
    total = sum(_message_tokens(m) for m in kept if isinstance(m, dict))
    if total > cap:
        logger.warning(
            "Honcho slimmed payload is still ~%d tokens > cap %d; passing through",
            total,
            cap,
        )
        return None

    slimmed = dict(request)
    slimmed["messages"] = kept
    return slimmed


def on_llm_request(*, request: dict, original_request: dict = None, **context: Any):
    """``llm_request`` middleware: drop the raw history Hermes replays.

    Hermes replays the entire in-session message list on every API call even
    though Honcho already observes each turn and injects its own context, so
    the same conversation is paid for twice and grows without bound. Keep only
    the leading system messages plus everything from the last user message
    onward — the current turn, including Honcho's injection and any in-flight
    tool-call/result pairs.

    Returns ``{"request": ...}`` to replace the payload, or None to leave it
    untouched. Fail-open on ANY uncertainty.
    """
    try:
        if context_fix_disabled():
            return None
        if not isinstance(request, dict):
            return None

        cap = max_tokens_cap()
        slimmed = _slim_messages(request, cap)
        if slimmed is None:
            return None

        original_messages = [m for m in request.get("messages", []) if isinstance(m, dict)]
        before = sum(_message_tokens(m) for m in original_messages)
        after = sum(_message_tokens(m) for m in slimmed["messages"] if isinstance(m, dict))
        dropped = len(request.get("messages", [])) - len(slimmed["messages"])
        logger.debug(
            "Honcho slimmed llm_request %d→%d msgs (~%d→~%d tokens, call #%s)",
            len(request.get("messages", [])),
            len(slimmed["messages"]),
            before,
            after,
            context.get("api_call_count"),
        )
        return {
            "request": slimmed,
            "source": "honcho-context-fix",
            "reason": (
                f"dropped {dropped} replayed messages; "
                "Honcho context is sole cross-turn memory"
            ),
        }
    except Exception as e:
        logger.warning("Honcho llm_request middleware error, passing through: %s", e)
        return None


def provider_injects_context(provider: Any) -> bool:
    """True when ``provider`` is in a mode/state where it injects context.

    Second gate in front of :func:`on_llm_request`, checking the provider's
    own state rather than the payload. It catches the cases the payload check
    cannot: with auth paused the only thing Honcho injects is the one-time
    "memory is paused" notice, which is fenced in ``<memory-context>`` like
    real context but replaces none of the history.

    Duck-typed so this module stays free of provider imports. An object whose
    state can't be read counts as injecting — the per-request check then has
    the final say, and both gates fail open.
    """
    try:
        if getattr(provider, "_cron_skipped", False):
            return False
        if getattr(provider, "_recall_mode", "hybrid") == "tools":
            return False
        if getattr(provider, "_init_auth_failure", None):
            return False
        manager = getattr(provider, "_manager", None)
        if manager is not None and getattr(manager, "_auth_failure", None):
            return False
        return True
    except Exception as e:
        logger.debug("Honcho middleware gate could not read provider state: %s", e)
        return True


def make_llm_request_middleware(provider: Any) -> Callable[..., Optional[dict]]:
    """Bind :func:`on_llm_request` to ``provider``'s recall mode and state.

    Registered instead of the bare function because the middleware callback
    is handed only the request and routing context, with no way back to the
    provider that decided whether to inject anything this run.
    """

    def _honcho_llm_request_middleware(**kwargs: Any) -> Optional[dict]:
        if not provider_injects_context(provider):
            return None
        return on_llm_request(**kwargs)

    return _honcho_llm_request_middleware
