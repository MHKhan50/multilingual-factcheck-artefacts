from civintel_agentbench.extraction import extract_features, normalize_for_matching


def test_normalization_preserves_meaningful_unicode():
    text = "  Гражданские  пострадали https://example.org/x "
    normalized = normalize_for_matching(text)
    assert "гражданские" in normalized
    assert "https://" not in normalized


def test_extracts_date_and_multilingual_event_cue():
    row = {"report_id": "x1", "language": "uk", "text": "Авіаудар стався 2026-01-09."}
    features = extract_features(row)
    assert features["date_mention"] == "2026-01-09"
    assert "airstrike" in features["event_type"]
