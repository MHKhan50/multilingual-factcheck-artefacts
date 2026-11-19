from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extraction import normalize_for_matching


def retrieve_pairs(records: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if len(records) < 2:
        return []
    texts = [normalize_for_matching(str(item.get("text", ""))) for item in records]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix)
    results: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        candidates = [
            (right_index, float(similarities[left_index, right_index]))
            for right_index in range(len(records)) if right_index != left_index
        ]
        candidates.sort(key=lambda item: (-item[1], str(records[item[0]].get("report_id", ""))))
        for right_index, score in candidates[:top_k]:
            if left_index < right_index:
                right = records[right_index]
                results.append({
                    "left_report_id": str(left.get("report_id", "")),
                    "right_report_id": str(right.get("report_id", "")),
                    "similarity": round(score, 6),
                    "same_language": str(left.get("language", "")) == str(right.get("language", "")),
                    "same_source_family": bool(
                        left.get("source_family") and
                        left.get("source_family") == right.get("source_family")
                    ),
                })
    unique = {}
    for pair in results:
        key = (pair["left_report_id"], pair["right_report_id"])
        unique[key] = pair
    return sorted(unique.values(), key=lambda item: (-item["similarity"], item["left_report_id"], item["right_report_id"]))
