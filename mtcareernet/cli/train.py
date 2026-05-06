"""CLI entry: `mtc-train --config-name default`."""
from __future__ import annotations
import argparse
from ..utils.logging import get_logger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("mtc-train")
    p.add_argument("--config-name", default="default")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260219)
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--overrides", nargs="*", default=[])
    args = p.parse_args(argv)
    log = get_logger("mtc.train")
    log.info("training entry: %s", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
