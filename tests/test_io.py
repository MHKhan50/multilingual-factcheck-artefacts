import pandas as pd
import pytest

from civintel_agentbench.io import load_reports


def test_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"report_id": "1", "text": "hello"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_reports(path)


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.csv"
    pd.DataFrame([
        {"report_id": "1", "source_name": "A", "language": "en", "published_at": "", "text": "a"},
        {"report_id": "1", "source_name": "B", "language": "en", "published_at": "", "text": "b"},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_reports(path)
