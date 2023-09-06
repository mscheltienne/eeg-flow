import argparse

from ..oddball import oddball


def run():
    """Run oddball() command."""
    parser = argparse.ArgumentParser(
        prog=f"{__package__.split('.')[0]}-oddball", description="oddball"
    )
    parser.add_argument(
        "condition", type=str, help="condition to run among 100 | 600 | 0a | 0b | a | b"
    )
    parser.add_argument(
        "--active",
        help="run the active oddball task",
        action="store_true",
    )
    parser.add_argument(
        "--dev",
        help="run with a mock trigger for dev purposes",
        action="store_true",
    )
    args = parser.parse_args()
    oddball(args.condition, args.active, args.dev)
