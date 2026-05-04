from __future__ import annotations

import sys

from main import main


if __name__ == "__main__":
    raise SystemExit(main(["--mode", "ai", *sys.argv[1:]]))