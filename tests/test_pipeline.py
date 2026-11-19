from pathlib import Path

from civintel_agentbench.pipeline import run_pipeline


def test_pipeline_writes_reproducibility_outputs(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "example_reports.csv"
    result = run_pipeline(str(input_path), str(tmp_path / "run"))
    assert result["input_rows"] == 5
    for filename in ["summary.json", "run_manifest.json", "report_pairs.csv", "evidence_audits.jsonl"]:
        assert (tmp_path / "run" / filename).exists()
