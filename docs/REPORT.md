# PickPresence Stage Report

## Current Milestone
- Name: M12 – GPU acceleration & provider selection
- Status: In progress

## Blockers
- M11 long-video stabilization is blocked by CPU-only throughput; M12 focuses on GPU acceleration to unblock performance targets.

## Deliverables Completed
- Added provider selection support (CLI/env) with explicit CUDA/CPU fallback logs.
- Added provider-selection unit tests.

## Verification
- Command: `./scripts/verify.sh`
- Result: Pass (pytest from system site-packages venv)

## End-to-End Acceptance (Clean Output Dir)
Run this from the repo root to validate M12 provider selection logic via unit tests:
```
./scripts/verify.sh
```

## Real-Video Acceptance (M12)
Provider detection (GPU preferred):
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"

GPU run (auto CUDA -> CPU fallback):
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python detectors/insightface_detector.py \
  --video /home/zhibai/videos/episode_full.mp4 \
  --output /home/zhibai/PickPresence/out/detections_gpu.json \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --sample-fps 2.0 \
  --max-frames 60

CPU run (force CPU provider):
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python detectors/insightface_detector.py \
  --video /home/zhibai/videos/episode_full.mp4 \
  --output /home/zhibai/PickPresence/out/detections_cpu.json \
  --providers CPUExecutionProvider \
  --sample-fps 2.0 \
  --max-frames 60

Expected: CUDA provider logs are preferred when available; otherwise logs include fallback to CPU.

## Performance Notes (M12)
- Provider probe (2026-02-02): `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.
- Timing run (2026-02-02, 180s video, `--sample-fps 1.0`):
  - GPU run (CUDAExecutionProvider active): 22.77s
  - CPU-only run: 103.98s
  - Speedup: ~4.57x (target ≥2x met)
- CUDA runtime stack installed (CUDA 12.9 + cuBLAS/cuRAND/cuFFT/cuDNN). Ensure `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.9/targets/x86_64-linux/lib` for CLI runs.

## Pressure Acceptance (M11)
Use the same episode video + references to validate expansion + trimming recovery (with chunking enabled):

1) Expansion run (relaxed thresholds + union, expect more segments):
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python -m pickpresence.cli \
  --video /home/zhibai/videos/episode.mp4 \
  --output-dir /home/zhibai/PickPresence/out_lab/m10_expand \
  --reference-dir /home/zhibai/PickPresence/out/refs \
  --reference-agg max \
  --detection-log /home/zhibai/PickPresence/out/detections.json \
  --segment-policy per_detection \
  --match-threshold 0.45 \
  --min-duration 0.6 \
  --merge-policy union \
  --export-end-eps 0.2 \
  --chunk-seconds 600 \
  --resume

## M10 Real-Video Acceptance (Confirmed)

### Run ID
out_audit/2026-01-26_1626

### Commands
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python -m pickpresence.cli \
  --video /home/zhibai/videos/episode.mp4 \
  --output-dir /home/zhibai/PickPresence/out \
  --reference-dir /home/zhibai/PickPresence/out/refs \
  --reference-agg topk_avg \
  --reference-topk 2 \
  --detection-log /home/zhibai/PickPresence/out/detections.json \
  --track-policy all \
  --min-track-similarity 0.5 \
  --merge-policy union \
  --trim-policy head_tail \
  --trim-source video \
  --trim-scan-window 0.6 \
  --trim-scan-step 0.04 \
  --bridge-gap 2.0 \
  --export-end-eps 0.2

### Outputs
- /home/zhibai/PickPresence/out/clip_000.mp4
- /home/zhibai/PickPresence/out/clip_001.mp4
- /home/zhibai/PickPresence/out/clip_002.mp4
- /home/zhibai/PickPresence/out/timeline.json

### Timeline Summary
- Segment 0: 2.0–5.2 (export 2.0–5.0)
- Segment 1: 10.4–25.6 (export 10.4–25.4)
- Segment 2: 28.6–32.4 (export 28.6–32.2)

### Manual Verification
All three clips contain the target (female lead) only; no male/other subjects observed.

### Conclusion
M10 multi-reference embeddings are validated on real video with improved recall for far/side-view shots while maintaining precision.

2) Trim recovery (tighten tail to remove male-only endings):
PYTHONPATH=/home/zhibai/PickPresence /home/zhibai/PickPresence/.venv-detector/bin/python -m pickpresence.cli \
  --video /home/zhibai/videos/episode.mp4 \
  --output-dir /home/zhibai/PickPresence/out_lab/m10_trimmed \
  --reference-dir /home/zhibai/PickPresence/out/refs \
  --reference-agg topk_avg \
  --reference-topk 2 \
  --detection-log /home/zhibai/PickPresence/out/detections.json \
  --segment-policy track_first \
  --merge-policy union \
  --trim-policy head_tail \
  --trim-source video \
  --trim-scan-window 0.6 \
  --trim-scan-step 0.04 \
  --trim-threshold-start 0.6 \
  --trim-threshold-keep 0.3 \
  --trim-min-run 3 \
  --trim-pad 0.2 \
  --export-end-eps 0.2 \
  --chunk-seconds 600 \
  --resume

## Manual Setup Instructions
1. **Detector dependencies** – Run `./scripts/setup_detector.sh` inside WSL/Linux to create `.venv-detector`, install `requirements-detector.txt` (InsightFace, onnxruntime, OpenCV), and verify ffmpeg availability. For GPU acceleration reinstall `onnxruntime-gpu` inside the venv.
2. **Environment config** – Copy `.env.example` to `.env`, then set:
   - `PICKPRESENCE_DETECTOR_SCRIPT=/abs/path/to/detectors/run_detector.sh`
   - `PICKPRESENCE_REFERENCE_EMBEDDING=/abs/path/to/<target>_embedding.json`
   - Optionally `PICKPRESENCE_OUTPUT_DIR=/abs/path/to/output`.
   - `CLI_PYTHON` defaults to `.venv-detector/bin/python`; set it explicitly only if you keep the CLI in another venv.
   - `PICKPRESENCE_CLI_ARGS="--reference-dir /abs/path/to/refs --reference-agg topk_avg --reference-topk 2 --segment-policy track_first --match-threshold-start 0.55 --match-threshold-keep 0.2 --track-policy all --min-track-similarity 0.0 --merge-policy union --export-end-eps 0.2 --trim-policy head_tail --trim-source video --trim-scan-window 0.6 --trim-scan-step 0.04 --trim-threshold-start 0.6 --trim-threshold-keep 0.3 --trim-min-run 3 --trim-pad 0.2 --min-duration 1.2 --bridge-gap 1.2"` bakes in the recommended multi-reference continuity + trimming knobs without editing the script (adjust thresholds per shot).
   - Demo/CLI automatically source `.env` (if present) before reading vars. Priority: CLI/demo arguments override exported shell vars, which override `.env` defaults.
3. **Reference embeddings** – Prefer the InsightFace backend:
   ```
   scripts/make_reference.py --backend insightface --name Alice \
     --image /path/alice.jpg --output /path/alice_ref.json
   ```
   (or `--video /path/video.mp4 --time 12.5`). Use `--backend toy --assume-face` only for tests.
4. **Detector models** – InsightFace automatically fetches SCRFD/ArcFace assets; keep them outside the repo (typically `~/.insightface`).
5. **Input video** – Provide an MP4/AVI clip; detector samples at 5 FPS by default. Use `PICKPRESENCE_DETECTOR_ARGS="--dump-frames 10 --dump-dir dump"` (or CLI flags) to inspect bounding boxes and similarities.
6. **Repo venv** – If pip access is blocked, create `.venv` with `python3 -m venv --system-site-packages .venv` so system `python3-pytest` can satisfy `./scripts/verify.sh`.

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

## M12 Postmortem (CUDA Enablement)
### What Went Wrong
- Provider list showed CUDA/TensorRT, but ONNXRuntime initially fell back to CPU because the venv had CPU-only `onnxruntime`.
- After switching to `onnxruntime-gpu`, CUDA provider failed to load due to missing runtime libs in sequence: `libcublasLt.so.12`, `libcurand.so.10`, `libcufft.so.11`, `libcudart.so.12`, then `libcudnn.so.9`.
- CUDA repo was pointed at `ubuntu2404` on a `ubuntu2204` system; apt refused due to missing GPG key.
- InsightFace model cache defaulted to `~/.insightface`, which was not writable in some runs.
- Codex sandbox could not access GPU devices even after CUDA was installed; GPU timing had to be run in the WSL terminal.

### Fixes Applied
- Added `PICKPRESENCE_INSIGHTFACE_ROOT` / `--model-root` to keep model cache under the repo.
- Installed CUDA keyring and switched CUDA apt source to `ubuntu2204`, importing the NVIDIA public key.
- Installed CUDA 12.9 runtime libs: cuBLAS, cuRAND, cuFFT, CUDA runtime, cuDNN.
- Ensured `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.9/targets/x86_64-linux/lib`.
- Verified CUDA EP via `onnxruntime` session creation in WSL, then re-ran GPU timing there.

### Outcome
- GPU timing (180s video, 1 FPS) improved from CPU 103.98s to GPU 22.77s (~4.57x).
- M12 acceptance now met; remaining GPU runs should be done in WSL terminal when required.
