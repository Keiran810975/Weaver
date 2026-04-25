import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import build_parser, run_experiment


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args, mode="single-node-multigpu")


if __name__ == "__main__":
    main()
