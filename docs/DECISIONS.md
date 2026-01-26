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
