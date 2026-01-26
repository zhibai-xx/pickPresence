#!/usr/bin/env python3
"""Combine multiple reference embeddings into a single template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pickpresence.embeddings import combine_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference template JSON.")
    parser.add_argument("--name", required=True, help="Name for the combined reference.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of reference JSON files to combine.",
    )
    parser.add_argument("--output", required=True, help="Path to write the template JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vectors = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vectors.append(data["embedding"])
    combined = combine_embeddings(vectors)
    payload = {"name": args.name, "embedding": combined}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[make_reference_template] Wrote template -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
