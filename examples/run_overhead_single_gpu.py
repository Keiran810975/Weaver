"""Convenience entrypoint for the single-GPU Weaver overhead experiment."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.run_overhead_experiment import main  # noqa: E402


DEFAULT_ARGS = ["--preset", "single_gpu_quick"]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
