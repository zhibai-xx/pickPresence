# PickPresence Stage Report

## Current Milestone
- Name: M9 – Clip-boundary stabilization
- Status: Ready for review

## Deliverables Completed
- Added `--export-end-eps` (default 0.20s) and switched ffmpeg to `-t duration`, so clips no longer include the final frame the timeline describes; timeline JSON now records `export_start/export_end`.
- Extended `--trim-policy head_tail` with `--trim-source {detections,video}` plus scan window/step knobs; the optional video path rescans frames via InsightFace to align head/tail with the last confirmed appearance.
- Segments expose `primary_track_id`, `match_avg`, `match_max`, and `match_p90` statistics; sources list whether trimming succeeded via detections or video and flag failures.
- Added `scripts/audit_segments.py` to dump `clip_last.png` and `orig_end_*` frames plus similarity PASS/FAIL summaries, making clip-boundary QA reproducible from a single command.

## Verification
- Command: `./scripts/verify.sh`
- Result: Pass (full CLI/tooling suite including the union + trim regressions)

## Manual Setup Instructions
1. **Detector dependencies** – Run `./scripts/setup_detector.sh` inside WSL/Linux to create `.venv-detector`, install `requirements-detector.txt` (InsightFace, onnxruntime, OpenCV), and verify ffmpeg availability. For GPU acceleration reinstall `onnxruntime-gpu` inside the venv.
2. **Environment config** – Copy `.env.example` to `.env`, then set:
   - `PICKPRESENCE_DETECTOR_SCRIPT=/abs/path/to/detectors/run_detector.sh`
   - `PICKPRESENCE_REFERENCE_EMBEDDING=/abs/path/to/<target>_embedding.json`
   - Optionally `PICKPRESENCE_OUTPUT_DIR=/abs/path/to/output`.
   - `CLI_PYTHON` defaults to `.venv-detector/bin/python`; set it explicitly only if you keep the CLI in another venv.
   - `PICKPRESENCE_CLI_ARGS="--segment-policy track_first --match-threshold-start 0.55 --match-threshold-keep 0.2 --track-policy all --min-track-similarity 0.0 --merge-policy union --export-end-eps 0.2 --trim-policy head_tail --trim-source detections --trim-threshold-start 0.6 --trim-threshold-keep 0.3 --trim-min-run 3 --trim-pad 0.2 --min-duration 1.2 --bridge-gap 1.2"` bakes in the recommended continuity + trimming knobs without editing the script (adjust `--trim-source video` + scan window when you want frame-level rescans).
   - Demo/CLI automatically source `.env` (if present) before reading vars. Priority: CLI/demo arguments override exported shell vars, which override `.env` defaults.
3. **Reference embeddings** – Prefer the InsightFace backend:
   ```
   scripts/make_reference.py --backend insightface --name Alice \
     --image /path/alice.jpg --output /path/alice_ref.json
   ```
   (or `--video /path/video.mp4 --time 12.5`). Use `--backend toy --assume-face` only for tests.
4. **Detector models** – InsightFace automatically fetches SCRFD/ArcFace assets; keep them outside the repo (typically `~/.insightface`).
5. **Input video** – Provide an MP4/AVI clip; detector samples at 5 FPS by default. Use `PICKPRESENCE_DETECTOR_ARGS="--dump-frames 10 --dump-dir dump"` (or CLI flags) to inspect bounding boxes and similarities.

## Demo Workflow
- Usage: `scripts/demo.sh /path/to/video.mp4 [/path/to/reference.json]`
  - Requires `PICKPRESENCE_DETECTOR_SCRIPT` (wrapper) and `PICKPRESENCE_REFERENCE_EMBEDDING` env vars, unless the reference path is provided as the second argument.
  - The script writes detections to `${PICKPRESENCE_OUTPUT_DIR:-./out}/detections.json`, runs `pickpresence.cli` to produce `timeline.json`, and exports clips (ffmpeg required).
  - Placeholder exports are disabled to ensure mp4 clip generation (libx264 + AAC). Install ffmpeg before running.
  - Supports InsightFace-specific args via `PICKPRESENCE_DETECTOR_ARGS` and CLI tuning via `PICKPRESENCE_CLI_ARGS`. Recommended process:
    1. Run with `--segment-policy per_detection --track-policy all --merge-policy union --export-end-eps 0.2 --bridge-gap 1.2` to confirm segments collapse from ~6 to ~3 (expect middle segment ~10.4–25.6s with `contrib_track_ids` like `[3,4,8,9]`).
    2. Re-run with trimming knobs enabled (e.g., `--trim-policy head_tail --trim-source detections --trim-threshold-start 0.6 --trim-threshold-keep 0.3 --trim-min-run 3 --trim-pad 0.2`, or switch to `--trim-source video --trim-scan-window 0.6 --trim-scan-step 0.04` for frame rescans). Export padding still applies.
  - Uses `.venv-detector/bin/python` (when available) plus exported `PYTHONPATH="$REPO"` so both detector and CLI share dependencies; override with `CLI_PYTHON` if needed.
  - The script auto-loads `.env` (if present) before reading defaults; passing a second argument or exporting variables in the shell still overrides `.env`.
  - On completion it prints the timeline path, clips directory, target name, segment count, and total duration.
  - Recommended manual acceptance:
    1. Run demo with union merge only and inspect `timeline.json` (expect ~3 segments; middle span covering ~10.4–25.6s with `contrib_track_ids` similar to `[3,4,8,9]`).
    2. Re-run with the trim flags enabled; verify the same middle segment tightens to ~10–24s (matching the last reliable detections) and exported clips remove male-only trailouts. Use `contrib_track_ids`, `primary_track_id`, and the new `match_*` stats to confirm which tracks contribute.
    3. Call `scripts/audit_segments.py --timeline out/timeline.json --video /path/to/video.mp4 --reference /path/to/ref.json --clips-dir out --output-dir out_lab/audit_images` to generate `clip_last.png`, the `orig_end_*` frames, and similarity PASS/FAIL results for each segment.

## Known Limitations
- InsightFace detector relies on downloaded model weights (not bundled); ensure they are cached locally before offline runs.
- Reference embeddings still need curated media; garbage-in/garbage-out remains true, though the insightface backend now shares logic with the detector. Provide multiple angles and merge via `scripts/make_reference_template.py` for best results.
- Demo assumes ffmpeg is installed; placeholder export is not enabled there.
- Union merging is purely time-based; if detectors emit overlapping tracks for different people within the same time span, manual review of `contrib_track_ids` is required.
- Head/tail trimming depends on detection fidelity. If detections disappear (e.g., occlusion) before the target actually leaves frame, the trimmed segment may end early; adjust `--trim-min-run`, thresholds, or disable trimming for those shots.
- Video-based trimming/audit scripts require InsightFace+OpenCV in the CLI environment; they run on CPU by default and may take a few seconds per segment when scanning dense windows.

## Next Steps
- Integrate ByteTrack/BoT-SORT or other dedicated trackers for longer-term stability and automatic drifting detection.
- Add scripts to capture reference embeddings directly from the target video (auto-cropping best face snapshots).
- Collect anonymized regression fixtures to automatically verify dump-frame outputs / ID drift once licensing allows sharing sample media.
