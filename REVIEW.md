# Code Review Rules

This file is the rubric the reviewer agent grades every PR against, and the
coder agent reads before writing code. Both agents must follow it exactly.
Edit this file to change how reviews behave in this repo — do not ask agents
to "just use best practices."

## Blocking rules (auto-reject — request changes)

1. **Security** — hardcoded secrets/API keys, shell injection, SQL injection,
   path traversal, `eval`/`exec` on user input, unauthenticated access to
   protected endpoints.
2. **Correctness** — logic contradicts the stated intent, off-by-one/race
   conditions, missing error handling on I/O/network/DB calls, code that
   cannot possibly do what the PR description claims.
3. **Tests** — new or changed behavior has no test, or the test suite does not
   pass. (If this repo has no test framework, say so in the review and grade
   Correctness manually.)

## Style / convention rules (warning-level, fix unless justified)

4. Match the repo's existing conventions: formatting, naming, error style,
   import order. Follow `AGENTS.md` / `CLAUDE.md` where present.
5. No dead code: no commented-out blocks, debug prints, or leftover TODOs in
   touched files.
6. Public functions/classes get docstrings/comments explaining *why*, not just
   *what*.
7. Changes are scoped to the task — no drive-by refactors of unrelated code.

## Approve criteria

Approve only when:
- No blocking rules are violated.
- Warnings are resolved or explicitly justified in a reply to the reviewer.
- The diff is small enough to fully review (if too large, request changes and
  ask for it to be split).

## Verdict format

The reviewer returns exactly one of:

- `approve` — no blocking issues; warnings resolved/justified.
- `request-changes` — at least one blocking issue or unaddressed warning,
  with a numbered list of findings: `file:line — severity — what's wrong —
  what to change`.

Do not approve with a caveat. If there is anything blocking, it is
`request-changes`.