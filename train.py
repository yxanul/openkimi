"""Compatibility entry point for ``k3-mini train``."""

import sys

from k3mini.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
