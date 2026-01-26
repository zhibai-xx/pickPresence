#!/usr/bin/env python3
"""Bootstrap pytest with warning suppression for argparse intermix parsing."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest


_original_showwarning = warnings.showwarning


def _showwarning(
    message,
    category,
    filename,
    lineno,
    file=None,
    line=None,
):
    text = str(message)
    if (
        category is UserWarning
        and "Do not expect file_or_dir" in text
        and "Namespace(" in text
    ):
        return
    return _original_showwarning(message, category, filename, lineno, file, line)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    warnings.showwarning = _showwarning
    return pytest.main()


if __name__ == "__main__":
    raise SystemExit(main())
