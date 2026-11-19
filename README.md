# CivIntel-AgentBench v0.1.0

**Multilingual Agentic AI for Conflict-Event Intelligence and Civilian-Harm Evidence Analysis**

CivIntel-AgentBench is a research prototype for evaluating how AI systems extract, retrieve, compare, and audit evidence about reported conflict-related incidents across English, Arabic, Russian, and Ukrainian sources.

## Research scope

The benchmark focuses on:
- extracting structured incident mentions from reports;
- retrieving and matching potentially related reports;
- comparing claims across independent sources;
- preserving source provenance and uncertainty;
- abstaining when evidence is insufficient.

It does **not** determine legal responsibility, establish that an alleged event occurred, classify a party as guilty, or replace professional investigations. A report is evidence that a source made a statement, not proof that the statement is true.

## v0.1.0 implementation

This release provides:
- a deterministic, CPU-friendly baseline pipeline;
- multilingual text normalization that preserves original text;
- transparent keyword/entity-field extraction for date, location, actor and event cues;
- TF-IDF retrieval and cosine-similarity report matching;
- a source-aware evidence audit with explicit uncertainty and abstention;
- synthetic fixture data for tests and a smoke run;
- JSONL/CSV input support, CLI, tests, and reproducibility metadata.

The baseline is deliberately modest. It is **not** a trained neural network or a production multi-agent system. It establishes an auditable evaluation harness onto which neural encoders and tool-using agents can be added and compared in later releases. This avoids claiming agentic or neural capabilities that have not yet been implemented.

## Quick start (macOS)

Python 3.9 or newer is required.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
civintel demo --output reports/demo
```

## Input format

CSV or JSONL files must contain:
- `report_id` (unique identifier)
- `source_name` (publisher or source label)
- `language` (`en`, `ar`, `ru`, `uk`, or another ISO language tag)
- `published_at` (ISO date/time string where available)
- `text` (original report text)

Optional fields: `url`, `event_date`, `location`, `actors`, `event_type`, `fatalities`, `source_family`.

Do not put personal data, graphic content, or unverified allegations about identifiable private individuals into test fixtures. Use public-interest reporting and follow source terms and licenses.

## CLI

```bash
civintel --help
civintel demo --output reports/demo
civintel evaluate --input data/example_reports.csv --output reports/evaluation
civintel inspect --input data/example_reports.csv --output reports/inspection.jsonl
```

`evaluate` reports retrieval and evidence-audit metrics against a small labelled pair file when supplied. Without labels, it emits descriptive diagnostics only; it will not fabricate ground-truth scores.

Optional pair-label CSV columns:
- `left_report_id`
- `right_report_id`
- `is_match` (0/1)

## Research questions

1. How does multilingual lexical retrieval compare with language-aware neural retrieval?
2. Does explicit source-family deduplication reduce false corroboration?
3. Can agentic evidence acquisition improve recall without increasing unsupported conclusions?
4. How well do confidence estimates correlate with actual evidence correctness?
5. When should the system abstain rather than issue a weakly supported match?

## Reproducibility

Each run writes:
- `summary.json`
- `report_pairs.csv`
- `evidence_audits.jsonl`
- `run_manifest.json` (timestamp, package version, input hash, configuration and environment)

The demo uses synthetic examples only. No current conflict facts or casualty counts are asserted.

## Ethical and methodological safeguards

- Distinguish reported claims from verified facts.
- Preserve source URLs and publication metadata.
- Track source-family identity to avoid counting syndication as independent corroboration.
- Do not infer identity, religion, ethnicity, intent, or legal culpability.
- Treat casualty figures as source-attributed claims, not ground truth.
- Include uncertainty and abstention in outputs.
- Human review is required before external use.

## Project layout

```text
CivIntel-AgentBench-v0.1.0/
├── data/
│   └── example_reports.csv
├── docs/
│   ├── DATA_SOURCES.md
│   ├── METHODOLOGY.md
│   └── ETHICS.md
├── reports/
├── src/civintel_agentbench/
│   ├── agents.py
│   ├── cli.py
│   ├── evaluation.py
│   ├── extraction.py
│   ├── io.py
│   ├── retrieval.py
│   └── utils.py
├── tests/
└── pyproject.toml
```

## Citation

If using this software in research, cite the repository release and the underlying datasets independently. Dataset access does not imply endorsement by dataset providers.
