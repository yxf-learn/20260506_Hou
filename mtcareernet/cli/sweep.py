import argparse
from ..utils.logging import get_logger


def main(argv=None) -> int:
    p = argparse.ArgumentParser("mtc-sweep")
    p.add_argument("--space", default="configs/sweeps/default.yaml")
    p.add_argument("--n-trials", type=int, default=32)
    p.add_argument("--budget-hours", type=float, default=8.0)
    args = p.parse_args(argv)
    get_logger("mtc.sweep").info("sweep entry: %s", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
