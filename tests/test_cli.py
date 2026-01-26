import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_generates_timeline_and_placeholder_clips(tmp_path):
    video = FIXTURES / "sample_video.txt"
    annotations = FIXTURES / "sample_annotations.json"
    output_dir = tmp_path / "artifacts"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--annotations",
        str(annotations),
        "--min-duration",
        "0.75",
        "--bridge-gap",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    timeline_path = output_dir / "timeline.json"
    assert timeline_path.exists()
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert data["video"].endswith("sample_video.txt")
    assert data["target"] == "unknown"
    assert data["summary"] == {
        "segment_count": 1,
        "total_duration": 2.4,
        "average_confidence": 0.8,
    }
    assert len(data["segments"]) == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] == 2.4
    assert seg["confidence"] == 0.8
    assert seg["sources"] == ["face-mock", "tracker-mock"]
    assert seg["track_id"] is None
    assert seg["primary_track_id"] is None
    assert seg["match_avg"] is None
    assert seg["match_max"] is None
    assert seg["match_p90"] is None
    tracks = data["tracks"]
    assert len(tracks) == 1
    track = tracks[0]
    assert track["track_id"] == "unknown"
    assert track["label"] is None
    assert track["avg_similarity"] is None
    assert track["p90_similarity"] is None
    assert track["max_similarity"] is None
    assert track["score"] is None
    assert track["total_duration"] == 2.4
    assert track["segment_count"] == 1
    assert track["detection_count"] is None

    clip_files = sorted(output_dir.glob("clip_*.txt"))
    assert len(clip_files) == 1
    clip_text = clip_files[0].read_text(encoding="utf-8")
    assert "placeholder clip for sample_video.txt" in clip_text
    assert "start=0.0, end=2.2" in clip_text or "start=0.0, end=2.199" in clip_text


def test_cli_uses_detection_log_and_reference_embedding(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "detections_target.json"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "detections_run"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--min-duration",
        "0.5",
        "--bridge-gap",
        "0.5",
        "--match-threshold",
        "0.75",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    timeline_path = output_dir / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert data["target"] == "SampleTarget"
    assert data["summary"]["segment_count"] == 1
    assert data["summary"]["total_duration"] == 1.6
    assert data["summary"]["average_confidence"] == 1.0
    assert len(data["segments"]) == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] == 1.6
    assert seg["confidence"] == 1.0
    assert "face-match" in seg["sources"]
    assert "track:1" in seg["sources"]
    assert seg["track_id"] == "1"
    assert seg["primary_track_id"] == "1"
    assert seg["match_avg"] == 1.0
    assert seg["match_max"] == 1.0
    assert seg["match_p90"] == 1.0
    assert len(data["tracks"]) == 1
    track = data["tracks"][0]
    assert track["track_id"] == "1"
    assert track["label"] == "SampleTarget"
    assert track["avg_similarity"] == 1.0
    assert track["p90_similarity"] == 1.0
    assert track["max_similarity"] == 1.0
    assert track["score"] == 1.0
    assert track["total_duration"] == 1.6
    assert track["segment_count"] == 1
    assert track["detection_count"] == 2


def test_export_end_eps_adjusts_timeline(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "detections_target.json"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "export_eps"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--match-threshold",
        "0.75",
        "--export-end-eps",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)
    timeline = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    seg = timeline["segments"][0]
    assert seg["export_start"] == seg["start"]
    assert pytest.approx(seg["export_end"], rel=1e-3) == seg["end"] - 0.5


def test_cli_selects_best_track_by_similarity(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "multi_detections.json"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "best_track"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--match-threshold",
        "0.6",
        "--min-track-duration",
        "0.5",
        "--bridge-gap",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert data["target"] == "SampleTarget"
    assert len(data["segments"]) == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0 and seg["end"] == 1.4
    assert seg["track_id"] == "A"
    assert {track["track_id"] for track in data["tracks"]} == {"A"}
    track = data["tracks"][0]
    assert track["segment_count"] == 1
    assert track["detection_count"] == 2
    assert track["score"] >= 0.6


def test_cli_track_policy_all_includes_multiple_tracks(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "multi_detections.json"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "all_tracks"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--target-name",
        "AllTargets",
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--track-policy",
        "all",
        "--min-track-similarity",
        "0.0",
        "--match-threshold",
        "0.6",
        "--min-duration",
        "0.4",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert data["target"] == "AllTargets"
    assert len(data["segments"]) == 2
    track_ids = {seg["track_id"] for seg in data["segments"]}
    assert track_ids == {"A", "B"}
    assert len(data["tracks"]) == 2
    assert {track["track_id"] for track in data["tracks"]} == {"A", "B"}
    for track in data["tracks"]:
        if track["track_id"] == "A":
            assert track["segment_count"] == 1
            assert track["detection_count"] == 2
            assert track["score"] >= 0.6
        if track["track_id"] == "B":
            assert track["segment_count"] == 1
            assert track["detection_count"] == 2


def test_cli_uses_ffmpeg_when_present(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "detections_target.json"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "ffmpeg_run"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ffmpeg = fake_bin / "ffmpeg"
    fake_ffmpeg.write_text(
        (FIXTURES / "fake_ffmpeg.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--match-threshold",
        "0.7",
    ]
    env = os.environ.copy()
    env.pop("PICKPRESENCE_FORCE_PLACEHOLDER", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    subprocess.run(cmd, check=True, env=env)

    clip_files = sorted(output_dir.glob("clip_*.mp4"))
    assert clip_files, "ffmpeg path should yield .mp4 artifacts"
    content = clip_files[0].read_text(encoding="utf-8")
    assert "FAKE FFMPEG OUTPUT" in content
    timeline = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["summary"]["segment_count"] == len(clip_files)


def test_track_first_policy_keeps_continuity(tmp_path):
    detections = FIXTURES / "track_first_detections.json"
    reference = FIXTURES / "reference_track.json"
    video = FIXTURES / "sample_video.txt"
    output_dir = tmp_path / "track_first"
    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--segment-policy",
        "track_first",
        "--min-duration",
        "0.2",
        "--bridge-gap",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)
    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] >= 1.5


def test_hysteresis_policy_keeps_segment(tmp_path):
    detections = FIXTURES / "track_first_detections.json"
    reference = FIXTURES / "reference_track.json"
    video = FIXTURES / "sample_video.txt"
    output_dir = tmp_path / "hysteresis"
    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections),
        "--segment-policy",
        "hysteresis",
        "--match-threshold-start",
        "0.7",
        "--match-threshold-keep",
        "0.2",
        "--min-duration",
        "0.2",
        "--bridge-gap",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)
    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] >= 1.5


def test_track_first_handles_low_average_similarity(tmp_path):
    video = FIXTURES / "sample_video.txt"
    reference = FIXTURES / "reference_track.json"
    detections_path = tmp_path / "low_avg.json"
    detections = [
        {"start": 0.0, "end": 0.5, "embedding": [2.0, 0.0, 0.0], "track_id": "edge", "sources": ["face-det"], "score": 0.9},
        {"start": 0.5, "end": 1.0, "embedding": [0.1, 1.0, 0.0], "track_id": "edge", "sources": ["face-det"], "score": 0.4},
        {"start": 1.0, "end": 1.5, "embedding": [0.05, 1.0, 0.0], "track_id": "edge", "sources": ["face-det"], "score": 0.3},
    ]
    detections_path.write_text(json.dumps(detections), encoding="utf-8")
    output_dir = tmp_path / "track_first_low_avg"
    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections_path),
        "--segment-policy",
        "track_first",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)
    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert data["summary"]["segment_count"] == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0 and seg["end"] == 1.5
    track = next(track for track in data["tracks"] if track["track_id"] == "edge")
    assert track["avg_similarity"] < 0.8
    assert track["score"] >= 0.8


def test_union_merge_policy(tmp_path):
    video = FIXTURES / "sample_video.txt"
    reference = FIXTURES / "reference_track.json"
    detections_path = tmp_path / "union.json"
    detections = [
        {"start": 0.0, "end": 1.0, "embedding": [1.0, 0.0, 0.0], "track_id": "3", "sources": ["face-det"], "score": 0.9},
        {"start": 1.05, "end": 2.0, "embedding": [0.98, 0.1, 0.0], "track_id": "4", "sources": ["face-det"], "score": 0.85},
        {"start": 2.1, "end": 3.0, "embedding": [0.99, 0.05, 0.0], "track_id": "6", "sources": ["face-det"], "score": 0.82},
        {"start": 3.05, "end": 4.0, "embedding": [0.97, 0.08, 0.0], "track_id": "7", "sources": ["face-det"], "score": 0.8},
    ]
    detections_path.write_text(json.dumps(detections), encoding="utf-8")

    base_cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections_path),
        "--segment-policy",
        "per_detection",
        "--match-threshold",
        "0.6",
        "--track-policy",
        "all",
        "--min-track-similarity",
        "0.0",
        "--min-duration",
        "0.2",
        "--bridge-gap",
        "0.5",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"

    out_no_union = tmp_path / "no_union"
    cmd_no_union = [*base_cmd, "--output-dir", str(out_no_union)]
    subprocess.run(cmd_no_union, check=True, env=env)
    timeline_no_union = json.loads((out_no_union / "timeline.json").read_text(encoding="utf-8"))
    assert timeline_no_union["summary"]["segment_count"] > 1

    out_union = tmp_path / "union"
    cmd_union = [*base_cmd, "--output-dir", str(out_union), "--merge-policy", "union"]
    subprocess.run(cmd_union, check=True, env=env)
    data = json.loads((out_union / "timeline.json").read_text(encoding="utf-8"))
    assert data["summary"]["segment_count"] == 1
    seg = data["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] == 4.0
    assert seg["track_id"] is None
    assert seg.get("contrib_track_ids") == ["3", "4", "6", "7"]
    assert "union-merge" in seg["sources"]


def test_trim_policy_head_tail(tmp_path):
    video = FIXTURES / "sample_video.txt"
    reference = FIXTURES / "reference_track.json"
    detections_path = tmp_path / "trim.json"
    detections = [
        {"start": 0.0, "end": 0.5, "embedding": [1.0, 0.0, 0.0], "track_id": "a", "sources": ["face-det"], "score": 0.95},
        {"start": 0.5, "end": 1.0, "embedding": [0.99, 0.05, 0.0], "track_id": "b", "sources": ["face-det"], "score": 0.9},
        {"start": 1.0, "end": 1.5, "embedding": [0.98, 0.1, 0.0], "track_id": "c", "sources": ["face-det"], "score": 0.88},
        {"start": 1.5, "end": 2.0, "embedding": [0.0, 1.0, 0.0], "track_id": "x", "sources": ["face-det"], "score": 0.2},
        {"start": 2.0, "end": 2.5, "embedding": [0.0, 1.0, 0.0], "track_id": "y", "sources": ["face-det"], "score": 0.2},
    ]
    detections_path.write_text(json.dumps(detections), encoding="utf-8")
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"

    base_cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--reference-embedding",
        str(reference),
        "--detection-log",
        str(detections_path),
        "--segment-policy",
        "per_detection",
        "--match-threshold",
        "0.0",
        "--track-policy",
        "all",
        "--min-track-similarity",
        "0.0",
        "--min-duration",
        "0.1",
        "--merge-policy",
        "union",
        "--force-placeholder-export",
    ]

    no_trim_dir = tmp_path / "trim_none"
    subprocess.run([*base_cmd, "--output-dir", str(no_trim_dir)], check=True, env=env)
    no_trim_timeline = json.loads((no_trim_dir / "timeline.json").read_text(encoding="utf-8"))
    assert no_trim_timeline["summary"]["segment_count"] == 1
    baseline_seg = no_trim_timeline["segments"][0]
    assert baseline_seg["end"] >= 2.4

    trim_dir = tmp_path / "trim_head_tail"
    trim_cmd = [
        *base_cmd,
        "--output-dir",
        str(trim_dir),
        "--trim-policy",
        "head_tail",
        "--trim-threshold-start",
        "0.8",
        "--trim-threshold-keep",
        "0.6",
        "--trim-min-run",
        "2",
        "--trim-pad",
        "0.1",
    ]
    subprocess.run(trim_cmd, check=True, env=env)
    trim_timeline = json.loads((trim_dir / "timeline.json").read_text(encoding="utf-8"))
    assert trim_timeline["summary"]["segment_count"] == 1
    trimmed_seg = trim_timeline["segments"][0]
    assert trimmed_seg["start"] == baseline_seg["start"]
    assert trimmed_seg["end"] < baseline_seg["end"]
    assert trimmed_seg["end"] <= 1.6


def test_cli_can_run_external_detector_script(tmp_path):
    video = FIXTURES / "sample_video.txt"
    reference = FIXTURES / "reference_target.json"
    output_dir = tmp_path / "detector_script_run"
    detector_output = tmp_path / "generated_detections.json"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--detector-script",
        str(FIXTURES / "detector_stub.py"),
        "--detector-output",
        str(detector_output),
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    assert detector_output.exists()
    # Should be identical to using detections_target fixture.
    timeline_path = output_dir / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert data["summary"]["segment_count"] == 1
    assert data["tracks"][0]["track_id"] == "1"


def test_cli_combines_reference_embeddings(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detection_path = tmp_path / "combo_detections.json"
    detection_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "embedding": [0.0, 1.0, 0.0],
                    "track_id": "A",
                    "sources": ["face-det"],
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    ref1 = tmp_path / "ref1.json"
    ref2 = tmp_path / "ref2.json"
    ref1.write_text(json.dumps({"name": "Front", "embedding": [1, 0, 0]}), encoding="utf-8")
    ref2.write_text(json.dumps({"name": "Side", "embedding": [0, 1, 0]}), encoding="utf-8")
    output_dir = tmp_path / "combo"
    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(ref1),
        "--reference-embeddings",
        str(ref2),
        "--detection-log",
        str(detection_path),
        "--segment-policy",
        "track_first",
        "--min-duration",
        "0.2",
        "--bridge-gap",
        "0.5",
        "--match-threshold",
        "0.6",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)
    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert data["summary"]["segment_count"] == 1


def test_cli_multi_reference_stats(tmp_path):
    video = FIXTURES / "sample_video.txt"
    detections = FIXTURES / "detections_target.json"
    reference = FIXTURES / "reference_target.json"
    ref_alt = tmp_path / "ref_alt.json"
    ref_alt.write_text(json.dumps({"name": "Side", "embedding": [0.0, 1.0, 0.0]}), encoding="utf-8")
    output_dir = tmp_path / "multi_ref"

    cmd = [
        sys.executable,
        "-m",
        "pickpresence.cli",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--reference-embedding",
        str(reference),
        "--reference-embeddings",
        str(ref_alt),
        "--reference-agg",
        "max",
        "--detection-log",
        str(detections),
        "--match-threshold",
        "0.7",
        "--force-placeholder-export",
    ]
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    subprocess.run(cmd, check=True, env=env)

    data = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    seg = data["segments"][0]
    assert seg["best_ref_id"] == "SampleTarget"
    assert seg["best_ref_sim"] >= 0.9
    track = data["tracks"][0]
    assert track["best_ref_id"] == "SampleTarget"
