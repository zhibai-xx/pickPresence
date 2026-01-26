# PickPresence Milestones & Acceptance

## M1 – Minimal CLI Pipeline (complete)
- [x] `pickpresence.cli` accepts `--video`, `--output-dir`, `--target-name` (optional) and produces a timeline JSON plus clipped artifacts.
- [x] Timeline generator implements identity confirmation stub, continuity bridge, and duration filtering knobs with documented defaults.
- [x] `scripts/verify.sh` runs tests that execute the CLI on `tests/fixtures/sample_video.txt` and assert artifacts exist with deterministic contents.
- [x] Placeholder clip export uses ffmpeg when available, otherwise the documented fallback, and both paths are covered by tests/mocks.

## M2 – Face-driven identification & tracking (complete)
- [x] CLI accepts reference embedding + detection log inputs (with match threshold) and plumbs them into the pipeline.
- [x] Timeline builder consumes detection entries, filters by cosine similarity, merges continuity gaps, and falls back to stub only when needed.
- [x] Regression tests cover the detection/reference path using fixtures to assert merged segments and metadata.
- [x] Documentation (DECISIONS/REPORT) reflects the new file formats and behavior for identity confirmation and tracking data.

## M3 – Production-ready pipeline (current)
- [x] Track-level clustering/selection policy (best/all) with similarity + duration thresholds exposes multi-identity workflows.
- [x] Timeline serialization includes `track_id` data and bridging respects per-track continuity to avoid cross-identity merges.
- [x] Tests cover best-track selection, multi-track export, and ffmpeg execution via a hermetic shim to validate binary invocation.
- [x] docs/REPORT.md captures the production-ready status plus limitations and follow-up guidance.

## M4 – Analytics & Reporting (complete)
- [x] Timeline JSON embeds summary stats (segment count, total duration, avg confidence) and per-track analytics to support downstream QA tooling.
- [x] CLI/tests validate the new metadata for annotation, detection, multi-track, and ffmpeg paths.
- [x] `scripts/verify.sh` remains the single entry point and now enforces the extended regression suite.
- [x] docs/REPORT.md reflects analytics readiness and next steps for integrating real model outputs into the reporting layer.

## M5 – Detector runner integration (complete)
- [x] CLI optionally runs an external detector script to generate detection logs before timeline building, with env-var defaults.
- [x] Regression tests cover the detector-runner path using a stub script that writes fixture detections.
- [x] Documented requirements for supplying real models/plugins plus guidance for manual installation/download steps.
- [x] Example configs (.env.example) and docs explain wiring actual detectors/videos.

## M6 – InsightFace detector v0 (complete)
- [x] `detectors/insightface_detector.py` integrates InsightFace SCRFD + ArcFace embedding with IoU + embedding gating to keep track IDs stable and avoid cross-identity drift.
- [x] Detector wrapper (`detectors/run_detector.sh`) launches `.venv-detector/bin/python` with the InsightFace detector and injects `PYTHONPATH` so imports resolve consistently.
- [x] `scripts/make_reference.py` supports an `insightface` backend (reusing the same embedding logic) alongside the lightweight toy backend for tests.
- [x] `scripts/demo.sh` and docs describe the InsightFace workflow, including dump-frame debugging and reference generation.
- [x] requirements + setup scripts provision InsightFace/onnxruntime inside `.venv-detector`, with documentation covering GPU/CPU notes.
- [x] Regression tests validate the detector wrapper contract and the reference generator; `./scripts/verify.sh` stays green.

## M7 – Continuity-aware segment policies (complete)
- [x] Track-first policy selects candidate tracks using percentile-based similarity scoring plus duration, then converts them into continuous segments regardless of per-frame similarity dips.
- [x] Hysteresis policy exposes `--match-threshold-start/keep` knobs to enter/maintain presence without fragmenting side-profile shots.
- [x] CLI/demo accept multi-reference templates so far-shot and profile embeddings can be merged into a single target vector.
- [x] Regression tests cover track-first continuity, hysteresis gating, and low-average-similarity cases; `./scripts/verify.sh` remains the single acceptance gate.
- [x] docs/REPORT.md documents recommended defaults (e.g., `--segment-policy track_first`, `--match-threshold-start/keep`) and the validation workflow.

## M8 – Cross-track union merge (complete)
- [x] Introduced `--merge-policy union` to post-process segments by time union regardless of track splits, honoring `bridge_gap` tolerances.
- [x] Union segments emit `contrib_track_ids` plus `union-merge` sources so reviewers know which track IDs were merged.
- [x] Regression tests validate that multiple tracks covering a continuous shot collapse into a single segment only when union merging is enabled.
- [x] docs/REPORT.md / `.env.example` capture the new flag, CLI usage, and recommended `PICKPRESENCE_CLI_ARGS` for demos.

## M9 – Clip-boundary stabilization (current)
- [x] Head/tail trim policy with hysteresis knobs (`--trim-threshold-start/keep`, `--trim-min-run`, `--trim-pad`, `--trim-source`) plus optional video scanning (window/step) keeps exported windows aligned with the last confirmed target frame.
- [x] Segments carry `match_avg/match_p90/match_max`, `primary_track_id`, and `contrib_track_ids`; union segments annotate their trimming source (`video-trim` / `det-trim`) and failures.
- [x] `--export-end-eps` subtracts a configurable safety pad and ffmpeg now uses `-t duration` so clip endings do not overrun. Timeline JSON records `export_start/export_end`.
- [x] `scripts/audit_segments.py` renders `clip_last.png` plus `orig_end_*` frames and prints similarity PASS/FAIL summaries for reproducible acceptance.
