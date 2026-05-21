import argparse
from ..utils.logging import get_logger


def main(argv=None) -> int:
    p = argparse.ArgumentParser("mtc-eval")
    p.add_argument("--run-id", required=False)
    p.add_argument("--ckpt", required=False)
    p.add_argument("--split", default="test", choices=["val", "cal", "test"])
    p.add_argument("--report", default="results/reports/last.json")
    args = p.parse_args(argv)
    get_logger("mtc.eval").info("eval entry: %s", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
