#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=False)
    args = parser.parse_args()

    # For tests we simply copy the canned detection fixture regardless of input.
    fixture = Path(__file__).with_name("detections_target.json")
    detections = json.loads(fixture.read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps(detections, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
