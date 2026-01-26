"""External detector runner utilities."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import MutableMapping, Sequence


def run_detector_script(
    script_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    reference_embedding: str | Path | None = None,
    extra_args: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Path:
    """Executes an external script/executable to produce a detection log."""

    command = _build_command(
        script_path=script_path,
        video_path=video_path,
        output_path=output_path,
        reference_embedding=reference_embedding,
        extra_args=extra_args,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=env)
    if not out_path.exists():
        raise RuntimeError(
            f"Detector script {script_path} completed but did not create {out_path}"
        )
    return out_path


def _build_command(
    script_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    reference_embedding: str | Path | None,
    extra_args: str | None,
) -> list[str]:
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Detector script not found: {script}")
    cmd = [str(script), "--video", str(video_path), "--output", str(output_path)]
    if reference_embedding:
        cmd.extend(["--reference", str(reference_embedding)])
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd
