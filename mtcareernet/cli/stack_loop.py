import argparse
from ..utils.logging import get_logger


def main(argv=None) -> int:
    p = argparse.ArgumentParser("mtc-stack-loop")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rrm-eta", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260219)
    args = p.parse_args(argv)
    get_logger("mtc.stack").info("stack-loop entry: %s", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
