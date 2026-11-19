from __future__ import annotations

from typing import Any

from .extraction import extract_features


def report_intelligence_agent(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent 1 baseline: deterministic incident-mention feature extraction."""
    return [extract_features(record) for record in records]


def evidence_retrieval_agent(pairs: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent 2 baseline: annotate retrieved pairs with source provenance."""
    by_id = {str(record["report_id"]): record for record in records}
    output = []
    for pair in pairs:
        left = by_id[pair["left_report_id"]]
        right = by_id[pair["right_report_id"]]
        output.append({
            **pair,
            "left_source": left.get("source_name", ""),
            "right_source": right.get("source_name", ""),
            "left_url": left.get("url", ""),
            "right_url": right.get("url", ""),
            "left_published_at": left.get("published_at", ""),
            "right_published_at": right.get("published_at", ""),
        })
    return output


def cross_source_analyst_agent(pairs: list[dict[str, Any]], threshold: float = 0.12) -> list[dict[str, Any]]:
    """Agent 3 baseline: conservative heuristic, not a factual verification decision."""
    audited = []
    for pair in pairs:
        independent = not pair.get("same_source_family", False)
        sufficient_similarity = float(pair.get("similarity", 0.0)) >= threshold
        audited.append({
            **pair,
            "candidate_related_reports": bool(independent and sufficient_similarity),
            "analyst_note": (
                "Potential cross-source match; requires human review."
                if independent and sufficient_similarity
                else "Insufficient basis for an independent corroboration claim."
            ),
        })
    return audited


def evidence_auditor_agent(pairs: list[dict[str, Any]], minimum_sources: int = 2) -> list[dict[str, Any]]:
    """Agent 4 baseline: records evidence sufficiency and abstains by default."""
    output = []
    for pair in pairs:
        has_two_sources = bool(pair.get("left_source")) and bool(pair.get("right_source")) and (
            pair.get("left_source") != pair.get("right_source")
        )
        candidate = bool(pair.get("candidate_related_reports")) and has_two_sources
        output.append({
            **pair,
            "distinct_source_count": 2 if has_two_sources else 1,
            "evidence_status": "candidate_for_review" if candidate else "insufficient_evidence",
            "decision": "abstain",
            "confidence": 0.0,
            "audit_warning": (
                "Similarity is not proof of event identity or truth; human verification required."
            ),
        })
    return output
