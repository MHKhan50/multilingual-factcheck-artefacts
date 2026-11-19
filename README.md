# CivIntel-AgentBench

[![tests](https://github.com/MHKhan50/CivIntel-AgentBench/actions/workflows/tests.yml/badge.svg)](https://github.com/MHKhan50/CivIntel-AgentBench/actions/workflows/tests.yml)

**Multilingual evidence-grounded claim verification and conflict-report analysis, with abstention.**

CivIntel-AgentBench evaluates how well systems verify claims about the Russia-Ukraine conflict from
evidence in English, Ukrainian, Russian and Chinese, when they should abstain, and whether they read
the evidence or shortcuts in it. It does not decide whether an event happened, attribute legal
responsibility, or replace professional fact-checking and investigation.

## v0.2.0 results on RU22Fact

Data: RU22Fact (Zeng et al., LREC-COLING 2024), 16,031 claims after dropping two malformed rows, with
the dataset's own splits (test: 1,741 English, 704 Ukrainian, 465 Russian and 305 Chinese claims).
Models: character n-gram TF-IDF with class-balanced logistic regression, regularisation chosen on
validation, test scored once. Macro-F1 over Supported, Refuted and NEI; per language over the labels
present (Ukrainian and Russian have no NEI claims); 95% bootstrap intervals over claims.

| Model input | Test macro-F1 [95% CI] | Accuracy | en | uk | ru | zh | Accuracy at 50% coverage |
|---|---|---|---|---|---|---|---|
| Majority (Supported) | 0.268 | 0.674 | | | | | |
| claim | 0.614 [0.594, 0.633] | 0.779 | 0.584 | 0.868 | 0.950 | 0.572 | 0.905 |
| evidence | 0.677 [0.655, 0.696] | 0.858 | 0.631 | 0.989 | 0.995 | 0.696 | 0.980 |
| claim+evidence | 0.679 [0.659, 0.698] | 0.859 | 0.631 | 0.989 | 0.995 | 0.713 | 0.981 |
| evidence format only (no words) | 0.591 [0.574, 0.609] | 0.761 | 0.543 | 0.944 | 0.931 | 0.352 | |
| claim+evidence, trained on English only | | | 0.641 | 0.236 | 0.927 | 0.503 | |

**Findings.**

1. **The evidence carries most of the signal, but much of it is format, not content.** A classifier
   that sees no words, only how the LLM-written evidence is formatted (a search-engine greeting,
   "[1]" citation markers, list style, length and character shares), reaches macro-F1 0.591, which
   is 79% of the gain of the full claim-and-evidence model over the majority baseline.
   In Russian, 95% of Supported but
   12% of Refuted evidence contains a search-engine
   greeting ("this is Bing"); in English, 83% of Supported and
   13% of Refuted evidence starts with "[1]"; in
   Ukrainian the median Supported evidence is 107 characters
   long against 2830 for Refuted.
   In Ukrainian and Russian the format alone reaches macro-F1
   0.944 and 0.931,
   close to the full model (0.989 and
   0.995).
   Scores on RU22Fact therefore overstate how well a model verifies claims.
2. **The claim alone already reaches macro-F1 0.614**, far above the majority
   baseline (0.268), so claims also carry label cues.
3. **NEI is the hard class** (F1 0.262 with claim and
   evidence).
4. **Abstention helps:** answering only the most confident half of the test claims raises accuracy
   from 0.859 to
   0.981.
5. **Lexical models do not transfer across scripts:** trained on English only, the model scores
   0.236 on Ukrainian; the higher
   Russian score comes from the format cues above, not from understanding Russian.

These are transparent baselines and an audit of the data, not a new verification model. Next:
multilingual neural encoders, artefact-controlled evaluation (evidence normalised to one format),
and tool-using agents that retrieve and cite evidence, evaluated on the same protocol.

Full numbers: [`results/ru22fact-v0.2.0/`](results/ru22fact-v0.2.0/).

## Quick start (macOS or Linux, Python 3.9+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
civintel ru22fact-download            # fetches RU22Fact from its pinned commit, checks SHA-256
civintel ru22fact-verify --output reports/ru22fact   # about 10 minutes on a laptop CPU
```

RU22Fact is not redistributed here: its repository has no licence file, so the download command
fetches it from the source for research use. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Report-matching pipeline (v0.1.0)

The original deterministic pipeline is kept: multilingual normalisation, transparent extraction of
date, place and actor cues, TF-IDF retrieval of candidate report pairs, and a source-aware audit that
does not count syndicated copies as independent corroboration. It runs on synthetic fixtures:

```bash
civintel demo --output reports/demo
civintel evaluate --input data/example_reports.csv --output reports/evaluation
```

## Safeguards

- A report or claim is evidence that someone said something, not proof that it is true.
- Labels are the fact-checkers' verdicts as released, not independently verified.
- No inference of identity, intent or legal responsibility; no use for targeting or surveillance.
- Human review is required before any external use. See [`docs/ETHICS.md`](docs/ETHICS.md).

## Project layout

```text
src/civintel_agentbench/
  ru22fact.py       download, checksum verification and loading of RU22Fact
  verification.py   baselines, bootstrap intervals, risk-coverage, transfer, format probe
  cli.py            civintel command
  pipeline.py, retrieval.py, extraction.py, agents.py, evaluation.py, io.py   v0.1.0 pipeline
results/ru22fact-v0.2.0/   results.json, results.md, run_manifest.json
docs/                      METHODOLOGY.md, DATA_SOURCES.md, ETHICS.md
tests/
```

## Citation

Cite this software with [`CITATION.cff`](CITATION.cff) and the data with:
Yirong Zeng, Xiao Ding, Yi Zhao, Xiangyu Li, Jie Zhang, Chao Yao, Ting Liu and Bing Qin. 2024.
RU22Fact: Optimizing Evidence for Multilingual Explainable Fact-Checking on Russia-Ukraine Conflict.
In *Proceedings of LREC-COLING 2024*, pages 14215-14226. <https://aclanthology.org/2024.lrec-main.1239>

## Licence

Code: MIT © 2026 Muzammal Hussain. Data: see the RU22Fact repository.
