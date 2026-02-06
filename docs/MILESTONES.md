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

## M9 – Clip-boundary stabilization (complete)
- [x] Head/tail trim policy with hysteresis knobs (`--trim-threshold-start/keep`, `--trim-min-run`, `--trim-pad`, `--trim-source`) plus optional video scanning (window/step) keeps exported windows aligned with the last confirmed target frame.
- [x] Segments carry `match_avg/match_p90/match_max`, `primary_track_id`, and `contrib_track_ids`; union segments annotate their trimming source (`video-trim` / `det-trim`) and failures.
- [x] `--export-end-eps` subtracts a configurable safety pad and ffmpeg now uses `-t duration` so clip endings do not overrun. Timeline JSON records `export_start/export_end`.
- [x] `scripts/audit_segments.py` renders `clip_last.png` plus `orig_end_*` frames and prints similarity PASS/FAIL summaries for reproducible acceptance.
- [x] docs/REPORT.md documents a clean-output end-to-end acceptance command and machine-readable summary for M9 verification.

## M10 – Multi-reference embeddings (complete)
- [x] FaceMatcher supports multi-reference aggregation (max/top-k avg) and preserves single-reference behavior.
- [x] CLI accepts multiple reference inputs (embeddings list + optional dir/list file) and passes aggregation knobs into the pipeline.
- [x] Timeline segments/tracks expose reference-side stats (`best_ref_id`, `best_ref_sim`, `best_ref_p90` or `ref_topk_avg`, `ref_hits`).
- [x] Tests cover aggregation modes + single-reference regression; `./scripts/verify.sh` remains green.
- [x] docs/REPORT.md documents M10 acceptance commands (clean fixtures + real video) and recommended CLI args.

## M11 – Long-video stabilization (current)
- [ ] Chunked processing with `--chunk-seconds` creates `out/chunks/segment_*/` with per-chunk detections + timelines.
- [ ] `--resume/--skip-existing` skips completed chunks and merges the remaining outputs.
- [ ] Merged output writes `out/timeline.json`, exports clips to `out/clips/`, and segments include `source_chunk`.
- [ ] Chunk performance stats (elapsed, FPS, skipped counts) are recorded in `out/chunks/summary.json` and summarized in docs/REPORT.md.
- [ ] Tests cover chunking + resume; `./scripts/verify.sh` remains green.

## M12 – GPU acceleration & provider selection (current)
- [x] Detector supports explicit provider order (CUDA/CPU) via CLI/env and logs selected providers.
- [x] CUDA unavailable triggers CPU fallback with explicit log entry.
- [x] Docs/.env.example include GPU configuration + acceptance commands (provider probe, CPU vs GPU timing).
- [x] Tests cover provider selection logic; `./scripts/verify.sh` remains green.
- [x] GPU vs CPU timing comparison recorded in docs/REPORT.md (≥2x speedup target on 120s chunk).

## M13 – Side-Profile Recall & Quality Policy (planned)
## M13 – Side-Profile Recall & Quality Policy (current)
- [x] Added side-profile recall knobs to handle mid/far side faces (including shadowed) without inflating default false positives.
- [x] Added explicit suppression for “small side-face” cases where target is not visual center (A/B/C).
- [x] Introduced segment quality tags (scale + lighting + side-profile ratio) in timeline output.
- [x] Added new policy knobs to separate “enter” vs “keep” for side profiles plus scale-aware gating.
- [x] Regression tests cover side-profile recall and small-face suppression; `./scripts/verify.sh` remains green.
- [x] docs/REPORT.md updated with M13 acceptance command and machine-readable summary output.

## M14 – Recall Recovery for Hard Side/Shadow Cases (current)
### M14.1 – Reference Expansion & Similarity Diagnostics (complete)
- [x] Add reference-set expansion workflow (side/low-light) with documented inputs/outputs.
- [x] Add similarity/track stability diagnostics for target gaps (summary JSON + optional images).
- [x] Provide an end-to-end acceptance command that rebuilds references and outputs diagnostics for S01E09_180.
- [x] Regression tests validate reference aggregation + diagnostics file schema; `./scripts/verify.sh` remains green.
- [x] docs/REPORT.md updated with how to run, expected outputs, and known limitations.

### M14.2 – Track Stability Bridge (Non-Face Fallback) (complete)
- [x] Integrate person-detection + tracking stub (CPU-safe default) to stabilize track IDs in far/side shots.
- [x] Add a continuity bridge that uses stable person tracks to fill face gaps (configurable, off by default).
- [x] Provide an acceptance command that outputs timeline + track stats for S01E09_180.
- [x] Regression tests cover track-stability bridge behavior and toggles; `./scripts/verify.sh` remains green.
- [x] docs/REPORT.md updated with metrics, validation command, and limitations.

### M14.3 – Optional ReID / Appearance Signal (Pluggable) (current)
- [x] Add optional appearance-based similarity (ReID or lightweight color/texture embedding) used only when face similarity is unreliable.
- [x] Gate appearance fallback with thresholds and log its contribution in timeline segments.
- [x] Provide a reproducible acceptance command demonstrating improved recall in hard segments.
- [x] Regression tests cover fallback gating and output metadata; `./scripts/verify.sh` remains green.
- [x] docs/REPORT.md updated with usage, expected gains, and trade-offs.

## M15 – Lightweight Person ReID Fallback (complete)
- [x] Add person-level embedding (downsampled body ROI) to detections and reference sets.
- [x] Add `--person-fallback` + `--person-threshold` to accept detections when face similarity is low.
- [x] Ensure track-fill and trim policies can use person similarity as a fallback gate.
- [x] Add track-fill caps (`--track-fill-max-duration`, `--track-fill-max-chain`) to prevent runaway merges.
- [x] Add face-confirm gating (`--face-confirm-threshold`, `--face-confirm-window`) to prevent cross-identity expansion.
- [x] Add regression tests covering person fallback path; `./scripts/verify.sh` remains green.
- [x] Update docs/REPORT.md with acceptance commands and expected outputs.
- [x] Add manual-review comparison script (timeline vs manual timestamps).

## M16 – Identity Consistency Upgrade (planned)
### M16.1 – Face-confirmed track expansion (complete data-driven spec)
- [ ] Define a formal “face-confirm anchor” rule for expansion (max gap, min confirmations).
- [ ] Add acceptance diff report against `docs/acceptance/S01E09_180_timeline_baseline.json`.
- [ ] Provide command + expected JSON summary in docs/REPORT.md.

### M16.2 – Person-consistency model (ReID/appearance) (planned)
- [ ] Evaluate a proper ReID embedding model or stronger appearance embedding (documented constraints).
- [ ] Add optional identity-consistency score to timeline (not used for gating by default).
- [ ] Regression tests for identity-consistency score schema and toggle behavior.

### M16.3 – Multi-signal identity gating (planned)
- [ ] Introduce configurable gating rules: face-confirmed OR (identity-consistency over window).
- [ ] Add guardrails to avoid cross-identity merges (max chain, min face confirmations per segment).
- [ ] Acceptance runs show improved coverage for 84–94 and 138–143 without introducing wrong-person clips.
