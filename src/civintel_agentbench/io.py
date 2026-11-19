from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"report_id", "source_name", "language", "published_at", "text"}


def load_reports(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        frame = pd.DataFrame(rows)
    else:
        raise ValueError("Input must be .csv or .jsonl")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["report_id"].duplicated().any():
        duplicates = frame.loc[frame["report_id"].duplicated(), "report_id"].tolist()
        raise ValueError(f"report_id values must be unique; duplicates: {duplicates[:5]}")
    frame = frame.fillna("")
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str)
    return frame
