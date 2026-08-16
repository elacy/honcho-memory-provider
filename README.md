# honcho-memory-provider

A user-installable, upgrade-surviving **[Honcho](https://docs.honcho.dev/v3/guides/integrations/hermes) memory provider for Hermes Agent.**

This is a **fork** of Hermes' bundled Honcho memory plugin
(`plugins/memory/honcho/`) that lives outside the Hermes source tree
(`$HERMES_HOME/plugins/<name>/`), so it can be modified freely and **survives
Hermes upgrades** — Hermes's provider discovery loads it as a distinct,
independently-named provider.

## Why a fork exists on disk but imports as a distinct name

Hermes loads memory providers by name from `plugins/memory/<name>/`, and
**bundled providers take precedence over user-installed ones on name
collisions** (this ordering is a deliberate invariant in Hermes' discovery
code). A copy that reuses the name `honcho` would therefore never load — the
bundled copy always wins. So this fork:

- lives under a distinct provider name (originally `honchofork`),
- rewrites all internal `plugins.memory.honcho.*` absolute imports to
  **relative** imports (`.`/`.client`/`.session`/`.oauth`/`.oauth_flow`/`.cli`),
- reports its provider `name` as the distinct name,
- still reads the same `$HERMES_HOME/honcho.json` config chain, so an existing
  workspace / peers carry over with no migration.

> ⚠️ **Name collision is a breaking invariant, not a suggestion.** Do **not**
> rename this back to `honcho` and drop it into a Hermes install — it will be
> silently shadowed by the bundled provider and never activate.

## What this fork adds over the bundled provider

The bundled Honcho provider has four defects that this fork addresses **directly
in the provider source** (rather than as separate runtime monkey-patches):

1. **Server-side token budget on `Session.context()`.** The bundled provider
   fetches session context uncapped and chops it client-side. This fork passes
   `tokens=` (from `contextTokens` in `honcho.json`) so the Honcho server builds
   the summary to fit.
2. **Synthesize a missing session summary.** Honcho only auto-generates a
   summary at message counts divisible by 20/60; per-thread Telegram sessions
   almost never reach that, so `ctx.summary` is null and task continuity is
   lost. This fork builds a compact transcript from `ctx.messages` when no
   summary exists.
3. **`llm_request` middleware that drops replayed raw history.** Hermes replays
   the full in-session message list on every API call even though Honcho already
   observes turns. A middleware keeps only the leading system block + the current
   turn, capped at ~40k tokens.
4. **`sync_turn` write/read race fix.** The write thread raced the next turn's
   context read; `sync_turn` now joins its write thread (bounded) before
   returning so the write commits first.

All four are **fail-open** (never risk dropping or corrupting the current turn)
and honor a `HONCHO_CONTEXT_FIX_DISABLE=1` kill switch.

> This fork originated as two artifacts that are now merged:
> - the forked provider itself (provider source, `docs/configuration.md`),
> - the `honcho-context-fix` monkey-patch plugin (the four fixes above, formerly
>   applied at runtime by wrapping the bundled provider's classes).

## Install

```bash
# clone into your Hermes user plugins dir
mkdir -p "$HERMES_HOME/plugins"
cp -r honcho-memory-provider "$HERMES_HOME/plugins/honchofork"

# select it as the active provider
hermes config set memory.provider honchofork
```

Set `contextTokens` in `$HERMES_HOME/honcho.json` if you want the server-side
budget (fix 1) active — the SDK patch is skipped when unset (uncapped remains
the explicit opt-out).

See [`docs/configuration.md`](docs/configuration.md) for the full provider
configuration reference (copied from the bundled Honcho plugin).

## Tools

Five bidirectional tools: `honcho_profile`, `honcho_search`,
`honcho_reasoning`, `honcho_context`, `honcho_conclude` (see the bundled
docs for details).

## Review

This repo is governed by [`REVIEW.md`](REVIEW.md); changes land as reviewed,
merged pull requests.