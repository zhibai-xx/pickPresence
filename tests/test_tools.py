import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from pickpresence.embeddings import EMBED_SIZE

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_detector_wrapper_is_executable():
    wrapper = ROOT_DIR / "detectors" / "run_detector.sh"
    assert wrapper.exists()
    assert os.access(wrapper, os.X_OK), "Wrapper must be executable"


def test_make_reference_produces_matching_embedding(tmp_path):
    image = tmp_path / "face.ppm"
    _write_ppm(image, width=4, height=4, color=(200, 120, 80))
    output = tmp_path / "ref.json"
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "make_reference.py"),
        "--name",
        "TestTarget",
        "--image",
        str(image),
        "--output",
        str(output),
        "--backend",
        "toy",
        "--assume-face",
    ]
    subprocess.run(cmd, check=True, env=env)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["name"] == "TestTarget"
    embedding = data["embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) == EMBED_SIZE
    assert all(isinstance(value, float) for value in embedding)


def _write_ppm(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes(color)
    data = pixel * width * height
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data)


def test_insightface_detector_mock(tmp_path):
    output = tmp_path / "det.json"
    mock = ROOT_DIR / "tests" / "fixtures" / "detections_target.json"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "detectors" / "insightface_detector.py"),
        "--video",
        str(mock),  # unused in mock mode
        "--output",
        str(output),
        "--mock-detections",
        str(mock),
    ]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["start"] == 0.0
    assert "embedding" in data[0]


def test_make_reference_template(tmp_path):
    ref1 = tmp_path / "ref1.json"
    ref2 = tmp_path / "ref2.json"
    ref1.write_text(json.dumps({"name": "A", "embedding": [1, 0, 0]}), encoding="utf-8")
    ref2.write_text(json.dumps({"name": "B", "embedding": [0, 1, 0]}), encoding="utf-8")
    output = tmp_path / "template.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "make_reference_template.py"),
            "--name",
            "Alice",
            "--inputs",
            str(ref1),
            str(ref2),
            "--output",
            str(output),
        ],
        check=True,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["name"] == "Alice"
    embedding = data["embedding"]
    assert len(embedding) == 3
    assert pytest.approx((embedding[0] ** 2 + embedding[1] ** 2) ** 0.5, rel=1e-3) == 1.0


def test_demo_loads_env_reference(tmp_path):
    env_path = ROOT_DIR / ".env"
    backup = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    reference = ROOT_DIR / "tests" / "fixtures" / "reference_target.json"
    detector_script = ROOT_DIR / "tests" / "fixtures" / "detector_stub.py"
    output_dir = tmp_path / "demo"
    sample_video = ROOT_DIR / "tests" / "fixtures" / "sample_video.txt"
    env_path.write_text(
        "\n".join(
            [
                f'PICKPRESENCE_REFERENCE_EMBEDDING="{reference}"',
                f'PICKPRESENCE_DETECTOR_SCRIPT="{detector_script}"',
                f'PICKPRESENCE_OUTPUT_DIR="{output_dir}"',
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PICKPRESENCE_FORCE_PLACEHOLDER"] = "1"
    try:
        subprocess.run(
            ["bash", "-c", f'cd "{ROOT_DIR}" && ./scripts/demo.sh "{sample_video}"'],
            check=True,
            env=env,
        )
    finally:
        if backup is None:
            env_path.unlink()
        else:
            env_path.write_text(backup, encoding="utf-8")

    timeline = output_dir / "timeline.json"
    assert timeline.exists()
    data = json.loads(timeline.read_text(encoding="utf-8"))
    assert data["segments"], "Demo should produce segments when env reference is loaded"


def test_demo_uses_detector_venv_python(tmp_path):
    env_path = ROOT_DIR / ".env"
    backup = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    env_path.write_text("", encoding="utf-8")
    fake_venv = tmp_path / "fakevenv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    log_file = tmp_path / "cli_env.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'echo "${{PYTHONPATH:-}}" > "{log_file}"',
                f'exec "{sys.executable}" "$@"',
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

    reference = ROOT_DIR / "tests" / "fixtures" / "reference_target.json"
    detector_script = ROOT_DIR / "tests" / "fixtures" / "detector_stub.py"
    output_dir = tmp_path / "demo_cli"
    sample_video = ROOT_DIR / "tests" / "fixtures" / "sample_video.txt"
    env = os.environ.copy()
    env.update(
        {
            "PICKPRESENCE_DETECTOR_SCRIPT": str(detector_script),
            "PICKPRESENCE_OUTPUT_DIR": str(output_dir),
            "PICKPRESENCE_REFERENCE_EMBEDDING": str(reference),
            "PICKPRESENCE_FORCE_PLACEHOLDER": "1",
            "DETECTOR_VENV_DIR": str(fake_venv),
        }
    )
    env.pop("CLI_PYTHON", None)

    try:
        subprocess.run(
            ["bash", "-c", f'cd "{ROOT_DIR}" && ./scripts/demo.sh "{sample_video}"'],
            check=True,
            env=env,
        )
    finally:
        if backup is None:
            env_path.unlink()
        else:
            env_path.write_text(backup, encoding="utf-8")

    timeline = output_dir / "timeline.json"
    assert timeline.exists()
    data = json.loads(timeline.read_text(encoding="utf-8"))
    assert data["segments"], "Demo should produce segments when detector stub is used"
    assert log_file.exists(), "CLI python stub should capture PYTHONPATH"
    logged = log_file.read_text(encoding="utf-8")
    assert str(ROOT_DIR) in logged, "Demo should export repository root to PYTHONPATH"
