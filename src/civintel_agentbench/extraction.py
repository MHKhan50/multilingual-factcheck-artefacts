from __future__ import annotations

import re
from typing import Any

DATE_PATTERNS = [
    re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(r"\b(19\d{2}[-/]\d{1,2}[-/]\d{1,2})\b"),
]
EVENT_TERMS = {
    "airstrike": ["airstrike", "air strike", "غارة جوية", "авиаудар", "авіаудар"],
    "shelling": ["shelling", "قصف", "обстрел", "обстріл"],
    "displacement": ["displacement", "displaced", "نزوح", "نازح", "перемещение", "переміщення"],
    "damage": ["damage", "destroyed", "تدمير", "دمار", "разрушен", "руйнув"],
    "civilian_casualty": ["civilian", "قتل مدني", "مدنيين", "гражданские", "цивільн"],
}
CONFLICT_TERMS = [
    "killed", "injured", "wounded", "civilian", "hospital", "school",
    "قتل", "جرح", "مدني", "مستشفى", "مدرسة",
    "погибли", "ранены", "гражданские", "больница", "школа",
    "загинули", "поранені", "цивільні", "лікарня", "школа",
]


def normalize_for_matching(text: str) -> str:
    """Conservative normalization; original text is always retained separately."""
    value = str(text).casefold()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_features(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text", ""))
    normalized = normalize_for_matching(text)
    date_match = next((pattern.search(text) for pattern in DATE_PATTERNS if pattern.search(text)), None)
    event_types = [
        label for label, terms in EVENT_TERMS.items()
        if any(term.casefold() in normalized for term in terms)
    ]
    cue_count = sum(term.casefold() in normalized for term in CONFLICT_TERMS)
    return {
        "report_id": str(row.get("report_id", "")),
        "language": str(row.get("language", "")),
        "date_mention": date_match.group(1) if date_match else "",
        "location": str(row.get("location", "")),
        "actors": str(row.get("actors", "")),
        "event_type": str(row.get("event_type", "")) or "|".join(event_types),
        "conflict_cue_count": int(cue_count),
        "text_length": len(text),
        "has_source_url": bool(str(row.get("url", "")).strip()),
        "extraction_method": "transparent_lexical_baseline",
    }
