# PickPresence Decisions & Assumptions

| ID | Date | Decision |
| --- | --- | --- |
| D1 | 2026-01-19 | For M1 we mock the identity pipeline with a deterministic placeholder that can be swapped for real detectors later. |
| D2 | 2026-01-19 | Clip export attempts to call ffmpeg if present; otherwise writes descriptor files to keep tests hermetic. |
| D3 | 2026-01-19 | `scripts/verify.sh` uses pytest as the single gate to keep CI parity with local development. |
| D4 | 2026-01-19 | Reference identities in M2 are stored as JSON files containing normalized embeddings to keep fixtures lightweight. |
| D5 | 2026-01-19 | Detection logs are JSON lists with bounding/embedding metadata; the CLI exposes `--detection-log`/`--reference-embedding` to feed them. |
| D6 | 2026-01-19 | Track selection aggregates detections by `track_id` with policy knobs (best/all) and thresholds for average similarity + total duration. |
| D7 | 2026-01-19 | ffmpeg execution is validated in tests using a hermetic shim injected via `PATH`, ensuring exporter logic remains testable without the real binary. |
| D8 | 2026-01-19 | Timelines embed summary + track-level analytics so downstream tooling can inspect durations/confidence without reprocessing clips. |
| D9 | 2026-01-19 | Detector invocation is scripted via `--detector-script`/env vars so real model pipelines can run externally and feed detections into the CLI. |
| D10 | 2026-01-19 | Detector execution is pinned to `.venv-detector` via `detectors/run_detector.sh` to avoid Python environment drift. |
| D11 | 2026-01-19 | Reference embeddings are generated through `scripts/make_reference.py`, sharing the same embedding logic as the detector for consistency. |
| D12 | 2026-01-19 | InsightFace SCRFD + ArcFace serves as the primary detector/embedding stack with IoU+embedding gating to avoid identity drift. |
| D13 | 2026-01-19 | `scripts/make_reference.py` defaults to a toy backend but supports InsightFace for production embeddings, keeping testability intact. |
| D14 | 2026-01-19 | Segment continuity favors track-first (percentile-scored) and hysteresis policies to keep side-profile shots intact; CLI exposes entry/keep knobs for tuning. |
| D15 | 2026-01-19 | Cross-track union merge is treated as an optional post-processing step (`--merge-policy union`) that unions contiguous segments regardless of track IDs while annotating contributing tracks. |
| D16 | 2026-01-19 | Head/tail trimming is an opt-in final pass that uses the reference matcher (hysteresis thresholds + run length + padding) to cut trailing frames where the target disappears, dropping segments that cannot be confirmed. |
| D17 | 2026-01-19 | Clip export always subtracts `--export-end-eps` and uses `-t duration` rather than `-to` to avoid ffmpeg’s last-frame drift; timeline records both logical and export ranges. |
| D18 | 2026-01-19 | Video-based trimming optionally rescans a configurable window via InsightFace to align head/tail boundaries with real frames; when unavailable it falls back to detection timestamps but annotates failures. |
| D19 | 2026-01-26 | M9 acceptance uses fixture-based synthetic detections and placeholder exports to keep the end-to-end verification deterministic without requiring ffmpeg or external media. |
| D20 | 2026-01-26 | Audit frame extraction adds ffmpeg `-update 1` and a single retry with size checks to stabilize single-image outputs without altering clip generation. |
| D21 | 2026-01-26 | Audit sampling offsets from clip end by 0.2s and writes ffmpeg stderr logs per frame to reduce EOF blank frames and aid debugging. |
| D22 | 2026-01-26 | Multi-reference matching uses per-reference similarity aggregation (default `max`, optional `topk_avg`) while keeping combined template output for detector compatibility. |
| D23 | 2026-01-26 | `scripts/verify_pytest.py` wraps pytest to suppress the Python 3.12 argparse intermix warning (`Do not expect file_or_dir...`) during verify runs since it originates from upstream parsing internals. |
| D24 | 2026-01-26 | Chunked long-video processing writes per-chunk outputs under `out/chunks/segment_*`, prefixes track IDs with chunk IDs on merge, and uses ffmpeg chunk extraction when a full detection log is unavailable. |
| D25 | 2026-01-26 | Chunk processing logs per-chunk start/end plus write completion, enforces detector output path per chunk, and applies a watchdog timeout (3x chunk duration) to avoid long-video stalls. |
| D26 | 2026-01-26 | `scripts/demo.sh` bypasses the full detector run when chunking is requested, and chunk detector stdout/stderr are captured per chunk for troubleshooting. |
| D27 | 2026-01-26 | GPU provider selection is controlled via a comma-separated provider list (CLI `--providers` or `PICKPRESENCE_PROVIDER_ORDER`), with CUDA fallback logging when unavailable. |
| D28 | 2026-02-01 | Provider probe shows CUDA/TensorRT providers; CUDA runtime libraries are installed, but ONNXRuntime reports no CUDA-capable device and falls back to CPU, blocking GPU speedup validation. |
| D29 | 2026-02-01 | Local dev `.venv` is created with `--system-site-packages` to reuse system `pytest` when pip access is blocked; this keeps `./scripts/verify.sh` runnable without external package downloads. |
| D30 | 2026-02-01 | InsightFace utilities accept `PICKPRESENCE_INSIGHTFACE_ROOT` / `--model-root` to keep model cache under the repo (avoids permission issues when default `~/.insightface` is not writable). |
| D31 | 2026-02-02 | CUDA provider requires CUDA 12 runtime libs (cuBLAS/cuRAND/cuFFT/cuDNN); add `/usr/local/cuda-12.9/targets/x86_64-linux/lib` to `LD_LIBRARY_PATH` for successful CUDAExecutionProvider use. |
| D32 | 2026-02-02 | M13 introduces side-profile thresholds with scale gating and small-face suppression; detection logs now record `frame_size` to enable scale-aware quality tagging in timeline outputs. |
| D33 | 2026-02-02 | Side-profile bridge merges adjacent segments when both have high `side_profile_ratio`, enabling local continuity fixes without global union merging. |
| D34 | 2026-02-02 | Detector recall tuning adds `--det-threshold` (lowered to 0.4 when needed) plus higher sample FPS to recover shadowed mid/far side profiles. |
| D35 | 2026-02-02 | Side-fill merges side-profile-heavy segments across larger gaps and filters segments dominated by small faces using `small-face-ratio-max`. |
| D36 | 2026-02-02 | Small-face filtering uses a triple gate (scale + low match + low side-profile ratio) to retain far side profiles while dropping tiny ambiguous faces. |
| D37 | 2026-02-02 | Video-based trimming can run on GPU via `--trim-device cuda` to reduce CPU saturation without changing trimming logic. |


## D38 Focus on mid/far side-face + low-light recall; de-prioritize small-side-face
- Date: 2026-02-04
- Context: Manual review shows major misses on mid/far side-face with shadow/low-light; small-side-face is a lower priority (may keep or drop).
- Decision: Treat small-side-face handling as optional/backlog. Primary optimization target is recall for mid/far side-face and shadowed side-face segments.
- Rationale: These misses are larger coverage gaps and higher impact than small-side-face false positives.
- Consequence: Parameter changes and heuristics should prioritize side-face recall even if small-side-face remains.


## D39 Lower side_scale_min to admit far side-faces into side-threshold path
- Date: 2026-02-04
- Context: Manual diff shows missed mid/far side-face segments with very small bbox area ratios (~0.0006–0.0013). Current side_scale_min=0.005 blocks side-threshold logic for these frames, leaving only strict match-threshold.
- Decision: In M13 tuning script, lower `--side-scale-min` to 0.001 to allow mid/far side-face into side-threshold scoring.
- Consequence: Expected recall increase for far side-faces; may increase false positives but small-face filtering remains optional.


## D40 Lower match/side thresholds to recover mid/far side-face in low-light
- Date: 2026-02-04
- Context: Lowering side_scale_min alone did not improve coverage; missed segments show detections present but filtered out by similarity thresholds.
- Decision: In M13 tuning script, set `--match-threshold 0.75`, `--side-threshold-start 0.50`, `--side-threshold-keep 0.40`.
- Consequence: Recall should increase on side-face/low-light; risk of extra false positives to be assessed with manual review.


## D41 Add track-fill continuity to recover side-face gaps
- Date: 2026-02-04
- Context: Lowering similarity thresholds did not recover mid/far side-face gaps; detections exist but segments remain discontinuous.
- Decision: Add a track-fill step that can insert detections inside short gaps when similarity is above a low threshold or track continuity applies. Exposed via `--track-fill-gap` and `--track-fill-min-similarity`.
- Consequence: May increase false positives; intended for M13 tuning focused on recall of side-face/low-light segments.


## D42 Track-fill now spans whole gap and merges across tracks
- Date: 2026-02-04
- Context: Initial track-fill inserted per-detection segments that were filtered by min_duration and failed to connect gaps.
- Decision: When a gap qualifies, insert a single gap-spanning segment (track_id=None) labeled `track-fill`, and allow bridge merges across tracks if either segment has `track-fill`.
- Consequence: Better continuity for side/low-light gaps; may merge across nearby tracks if gap is too large.


## D43 Limit BLAS/OMP threads in M13 script to reduce CPU heat
- Date: 2026-02-04
- Context: User observed high CPU core temperatures during pipeline runs.
- Decision: Set OMP/BLAS thread envs in `scripts/run_m13_style_180.sh` with defaults to 4 (override via env).
- Consequence: Lower peak CPU usage/heat; potential small slowdown mitigated by GPU usage.


## D44 Track-fill falls back to similarity-gated detections when track continuity is missing
- Date: 2026-02-04
- Context: Track IDs fragment in mid/far side-face segments; requiring same-track detections inside gaps prevents fills.
- Decision: In track-fill, prefer detections from adjacent track IDs, but fall back to similarity-gated detections if none exist.
- Consequence: Recovers gaps even when tracker re-IDs; slight risk of cross-identity fills when similarity gate is too low.


## D45 Skip trim for track-fill segments to preserve gap continuity
- Date: 2026-02-04
- Context: Track-fill segments were dropped by head/tail trimming because similarity thresholds remain high in side/low-light gaps.
- Decision: When a segment is labeled `track-fill`, bypass trim and keep the full gap span (marked `track-fill-keep`).
- Consequence: Retains low-confidence gap spans; may include more noise but restores continuity.


## D46 Move track-fill to post-trim stage
- Date: 2026-02-04
- Context: Track-first produces dense segments before trim; gaps only appear after trimming, so pre-trim track-fill had no effect.
- Decision: Apply track-fill after trimming to bridge gaps in the final timeline.
- Consequence: Track-fill now targets true recall gaps; may slightly alter final merging behavior.


## D47 Lower track-fill similarity gate to 0.20 for hard gaps
- Date: 2026-02-04
- Context: Remaining hard gaps (84–94, 138–152) not filled at 0.25 similarity gate.
- Decision: Reduce `--track-fill-min-similarity` to 0.20 for M13 tuning.
- Consequence: Higher recall in gap fill, slightly higher false-positive risk.


## D48 Add segment inspection helper for manual review
- Date: 2026-02-04
- Context: Need visual inspection for problematic mid/far side-face segments with similarity overlays.
- Decision: Add `scripts/inspect_segments.sh` and `scripts/inspect_segments.py` plus a default segment list for S01E09_180.
- Consequence: Produces clipped segments, per-frame detections JSON, and extracted frames for manual review.


## D49 Add batch reference set builder for side/low-light expansion
- Date: 2026-02-04
- Context: Need a repeatable way to build larger reference sets (side/low-light) without manual one-off commands.
- Decision: Add `scripts/make_reference_set.py` and `scripts/build_reference_set.sh` to batch-generate reference JSONs and an index file.
- Consequence: Reference expansion becomes reproducible and can be used in acceptance workflows.

## D50 Add similarity/track diagnostics script for gap analysis
- Date: 2026-02-04
- Context: Need quantitative diagnostics (similarity distribution + track stability) for hard segments.
- Decision: Add `scripts/diagnose_segments.py` + wrapper to emit a summary JSON from detection logs and reference embeddings.
- Consequence: Enables repeatable gap analysis without manual inspection.


## D51 Add track stabilization stub (similarity + gap)
- Date: 2026-02-04
- Context: Diagnostics show heavy track switching in mid/far side-face segments.
- Decision: Add `--track-stabilize` with gap + similarity thresholds to reassign fragmented track IDs into stable IDs before track selection.
- Consequence: Better continuity for far/side shots; still face-similarity based (not full person ReID).


## D52 Add appearance-fallback matching for low-face-sim cases
- Date: 2026-02-04
- Context: Hard segments still have low face similarity but visible targets; need a non-face signal.
- Decision: Add appearance histogram embeddings to references/detections and allow appearance similarity to pass thresholds when face similarity is low (optional, default off).
- Consequence: Improves recall in far/side/occluded shots; risk of false positives if appearance is ambiguous.


## D53 Add lightweight person ReID fallback (downsampled body embedding)
- Date: 2026-02-05
- Context: Appearance histograms helped but still miss far/side cases; need a stronger non-face cue without heavy models.
- Decision: Add a lightweight person embedding (downsampled ROI) to detections/reference sets and allow `--person-fallback` to pass thresholds when face similarity is low.
- Consequence: Improves recall in hard side-profile segments; risk of cross-identity confusion if clothing/body cues are similar.


## D54 Add track-fill caps to prevent runaway merges
- Date: 2026-02-05
- Context: Track-fill can chain-bridge many short gaps into very long segments in long sequences.
- Decision: Add `--track-fill-max-duration` (per-gap cap) and `--track-fill-max-chain` (max track-fill merge chain) to limit over-merge.
- Consequence: Limits excessive continuity; may reduce recall when true gaps are long but helps avoid giant false merges.


## D55 Add clean-output option to avoid stale clips
- Date: 2026-02-05
- Context: Re-running the pipeline left old clip files in place, confusing manual review.
- Decision: Add `--clean-output` to remove existing `clip_*` and `timeline.json` before export; enable it in the M15 run script by default.
- Consequence: Avoids stale artifacts but requires re-exporting all clips each run.


## D56 Add face-confirm gating to prevent non-target expansion
- Date: 2026-02-05
- Context: Person/appearance fallback caused cross-identity expansion in shared scenes.
- Decision: Add face-confirm gating (threshold + window) so side-bridge and track-fill only expand segments if a high-confidence face confirmation exists near segment edges.
- Consequence: Preserves identity precision while still allowing limited side/track expansion around verified faces.


## D57 Add manual-review comparison script
- Date: 2026-02-05
- Context: Manual review timestamps need systematic comparison against generated timelines.
- Decision: Add `scripts/compare_manual_review.py` + wrapper to compute coverage ratios and overlap pairs.
- Consequence: Enables repeatable acceptance checks against manual review data.


## D58 Lock M15 tuning parameters for accepted timeline
- Date: 2026-02-05
- Context: Manual review confirmed near-acceptable output with only minor, acceptable misses.
- Decision: Lock the M15 accepted parameters (track-stabilize, trim thresholds, side thresholds, track-fill limits, union merge) in docs/REPORT.md for reproducible runs.
- Consequence: Provides a stable baseline for M16 work; future changes must justify deviations.


## D59 Start M16 identity-consistency upgrade
- Date: 2026-02-05
- Context: Remaining misses are in far/side/low-light segments; threshold tuning alone is insufficient.
- Decision: Plan M16 to introduce identity-consistency signals (face-confirm anchoring, optional ReID/appearance scoring) with strict guardrails.
- Consequence: Shifts focus from tuning to structured identity consistency, aiming to improve recall without new false positives.
