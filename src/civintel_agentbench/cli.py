from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="civintel",
        description="Evidence-grounded multilingual conflict-report research benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the synthetic smoke-test dataset.")
    demo.add_argument("--output", default="reports/demo")

    evaluate = subparsers.add_parser("evaluate", help="Run the pipeline on CSV or JSONL reports.")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--labels", default=None)
    evaluate.add_argument("--top-k", type=int, default=3)

    inspect = subparsers.add_parser("inspect", help="Run evidence extraction and audit on a dataset.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        input_path = str(Path(__file__).resolve().parents[2] / "data" / "example_reports.csv")
        result = run_pipeline(input_path, args.output)
    elif args.command == "evaluate":
        result = run_pipeline(args.input, args.output, args.top_k, args.labels)
    else:
        result = run_pipeline(args.input, args.output, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
