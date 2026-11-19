from civintel_agentbench.retrieval import retrieve_pairs


def test_retrieval_returns_deterministic_unique_pairs():
    records = [
        {"report_id": "a", "language": "en", "text": "hospital damaged after airstrike"},
        {"report_id": "b", "language": "en", "text": "hospital damaged after airstrike"},
        {"report_id": "c", "language": "ru", "text": "different unrelated topic"},
    ]
    first = retrieve_pairs(records, top_k=2)
    second = retrieve_pairs(records, top_k=2)
    assert first == second
    assert len({(p["left_report_id"], p["right_report_id"]) for p in first}) == len(first)


def test_retrieval_validates_top_k():
    import pytest
    with pytest.raises(ValueError):
        retrieve_pairs([], top_k=0)
