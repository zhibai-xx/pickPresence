#!/usr/bin/env python3
import sys
from pathlib import Path


def main(argv):
    if "-to" in argv:
        end = argv[argv.index("-to") + 1]
    else:
        end = "unknown"
    if "-ss" in argv:
        start = argv[argv.index("-ss") + 1]
    else:
        start = "unknown"
    output = Path(argv[-1])
    output.write_text(
        f"FAKE FFMPEG OUTPUT\nstart={start}\nend={end}\nargs={' '.join(argv)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
