# PickPresence Autonomous Agent Agreement

## Mission
- Deliver a local-first pipeline that ingests a video plus target identity reference and emits filtered timelines and clipped segments.
- Maintain a closed-loop workflow: plan, implement, test via `scripts/verify.sh`, and update documentation without relying on the user for micro-confirmations.

## Operating Rules
- `scripts/verify.sh` is the only supported gate. Every change must keep it green; fix regressions immediately.
- Never remove user-authored work or destructive commands unless explicitly asked.
- All assumptions or trade-offs must be logged in `docs/DECISIONS.md`.
- Milestones, acceptance criteria, and progress tracking live in `docs/MILESTONES.md`.
- Stage reports go to `docs/REPORT.md` after each milestone.
- Keep credentials/configuration out of source; use environment variables and `.env.example`.

## Execution Loop
1. Read instructions and repository state.
2. Create/update plan; implement features with tests/fixtures.
3. Run `./scripts/verify.sh` locally; ensure exit code 0.
4. Update docs (MILESTONES, DECISIONS, REPORT) to stay consistent with the delivered code.
5. Present concise status plus next action; block only when necessary and provide decision options.

## Communication
- Default to autonomous execution; only ask for input if blocked.
- Reports must focus on current milestone readiness and verifiable outputs.

## Tooling
- Language: Python CLI for now, extendable.
- Tests: `pytest` plus fixtures under `tests/fixtures/`.
- Video processing: prefer ffmpeg; allow placeholder implementation when the binary is unavailable, but record rationale in `docs/DECISIONS.md`.
