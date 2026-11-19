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

    dl = subparsers.add_parser("ru22fact-download", help="Download RU22Fact from its pinned commit and verify it.")
    dl.add_argument("--data-dir", default="data/raw/ru22fact")

    verify = subparsers.add_parser("ru22fact-verify", help="Claim-verification baselines on RU22Fact.")
    verify.add_argument("--data-dir", default="data/raw/ru22fact")
    verify.add_argument("--output", default="reports/ru22fact")
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--bootstrap", type=int, default=1000)

    neural = subparsers.add_parser("ru22fact-neural", help="Fine-tune XLM-R on raw and artefact-removed evidence (GPU).")
    neural.add_argument("--data-dir", default="data/raw/ru22fact")
    neural.add_argument("--output", default="reports/ru22fact-neural")
    neural.add_argument("--model", default="xlm-roberta-base")
    neural.add_argument("--seeds", default="0,1,2")
    neural.add_argument("--variants", default="raw,cleaned")
    neural.add_argument("--max-len", type=int, default=256)
    neural.add_argument("--epochs", type=int, default=3)
    neural.add_argument("--limit", type=int, default=None, help="Claims per label and split, for a quick smoke run.")
    return parser


def run_ru22fact(data_dir: str, output: str, seed: int, bootstrap: int) -> dict:
    from . import __version__
    from .ru22fact import COMMIT, REPOSITORY, load
    from .utils import environment_info, utc_now, write_json
    from .verification import results_table, run_verification

    data = load(data_dir)
    results = run_verification(data, seed=seed, bootstrap=bootstrap)
    out = Path(output)
    write_json(out / "results.json", results)
    (out / "results.md").write_text(
        "# RU22Fact claim-verification baselines\n\nTest split, macro-F1 over Supported, Refuted and NEI; "
        "per language over the labels present (Ukrainian and Russian have no NEI claims); "
        "95% bootstrap interval over claims.\n\n" + results_table(results) + "\n", encoding="utf-8")
    write_json(out / "run_manifest.json", {
        "package_version": __version__, "created_utc": utc_now(), "seed": seed,
        "dataset": REPOSITORY, "dataset_commit": COMMIT,
        "dropped_malformed_rows": data.attrs.get("dropped_malformed", 0), **environment_info(),
    })
    return {"output": str(out), "test_macro_f1": {k: results[k]["macro_f1"] for k in ("majority", "claim", "evidence", "claim+evidence")}}


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        input_path = str(Path(__file__).resolve().parents[2] / "data" / "example_reports.csv")
        result = run_pipeline(input_path, args.output)
    elif args.command == "evaluate":
        result = run_pipeline(args.input, args.output, args.top_k, args.labels)
    elif args.command == "ru22fact-download":
        from .ru22fact import download

        result = {"data_dir": args.data_dir, "sha256": download(args.data_dir)}
    elif args.command == "ru22fact-neural":
        from .neural import run_neural
        from .ru22fact import load

        result = run_neural(load(args.data_dir), args.output, seeds=[int(x) for x in args.seeds.split(",")],
                            variants=args.variants.split(","), model_name=args.model, max_len=args.max_len,
                            epochs=args.epochs, limit=args.limit)
    elif args.command == "ru22fact-verify":
        result = run_ru22fact(args.data_dir, args.output, args.seed, args.bootstrap)
    else:
        result = run_pipeline(args.input, args.output, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
