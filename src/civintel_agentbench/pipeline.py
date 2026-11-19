from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .agents import (
    cross_source_analyst_agent,
    evidence_auditor_agent,
    evidence_retrieval_agent,
    report_intelligence_agent,
)
from .evaluation import evaluate_pairs
from .retrieval import retrieve_pairs
from .utils import environment_info, sha256_file, utc_now, write_json


def run_pipeline(input_path: str, output_dir: str, top_k: int = 3, labels_path: str | None = None) -> dict[str, Any]:
    from .io import load_reports

    frame = load_reports(input_path)
    records = frame.to_dict("records")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    extracted = report_intelligence_agent(records)
    retrieved = retrieve_pairs(records, top_k=top_k)
    sourced = evidence_retrieval_agent(retrieved, records)
    analysed = cross_source_analyst_agent(sourced)
    audited = evidence_auditor_agent(analysed)

    pd.DataFrame(extracted).to_csv(output / "extracted_features.csv", index=False)
    pd.DataFrame(audited).to_csv(output / "report_pairs.csv", index=False)
    with (output / "evidence_audits.jsonl").open("w", encoding="utf-8") as handle:
        for item in audited:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = evaluate_pairs(audited, labels_path)
    summary = {
        "project": "CivIntel-AgentBench",
        "version": __version__,
        "created_at_utc": utc_now(),
        "input_rows": len(records),
        "retrieved_pair_count": len(audited),
        "candidate_related_pair_count": sum(bool(item["candidate_related_reports"]) for item in audited),
        "abstention_count": sum(item["decision"] == "abstain" for item in audited),
        "metrics": metrics,
        "limitations": [
            "Lexical baseline only; no neural model is trained in v0.1.0.",
            "Agent stages are deterministic functions, not autonomous LLM agents.",
            "Similarity is not factual verification.",
            "No legal attribution or casualty verification is performed.",
        ],
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "project": "CivIntel-AgentBench",
        "version": __version__,
        "created_at_utc": summary["created_at_utc"],
        "input_path": str(Path(input_path)),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(records),
        "top_k": top_k,
        "labels_path": labels_path or None,
        "environment": environment_info(),
        "pipeline": [
            "report_intelligence_agent",
            "evidence_retrieval_agent",
            "cross_source_analyst_agent",
            "evidence_auditor_agent",
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    return summary
