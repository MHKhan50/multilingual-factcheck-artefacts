from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def evaluate_pairs(predictions: list[dict[str, Any]], labels_path: str | None = None) -> dict[str, Any]:
    if not labels_path:
        return {
            "evaluation_mode": "descriptive_only",
            "labelled_pair_metrics": None,
            "note": "No labelled pair file supplied; no ground-truth performance scores are reported.",
            "retrieved_pair_count": len(predictions),
        }
    labels = pd.read_csv(labels_path)
    required = {"left_report_id", "right_report_id", "is_match"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label file missing columns: {sorted(missing)}")
    pred_map = {
        (item["left_report_id"], item["right_report_id"]): int(item.get("candidate_related_reports", False))
        for item in predictions
    }
    y_true, y_pred = [], []
    for row in labels.to_dict("records"):
        key = (str(row["left_report_id"]), str(row["right_report_id"]))
        reverse = (key[1], key[0])
        y_true.append(int(row["is_match"]))
        y_pred.append(pred_map.get(key, pred_map.get(reverse, 0)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "evaluation_mode": "labelled_pairs",
        "labelled_pair_metrics": {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "n_pairs": len(y_true),
        },
        "retrieved_pair_count": len(predictions),
    }
