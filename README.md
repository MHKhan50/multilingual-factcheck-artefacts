# Multilingual fact-check artefacts (CivIntel-AgentBench)

[![tests](https://github.com/MHKhan50/multilingual-factcheck-artefacts/actions/workflows/tests.yml/badge.svg)](https://github.com/MHKhan50/multilingual-factcheck-artefacts/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24370943.svg)](https://doi.org/10.5281/zenodo.24370943)

**Multilingual evidence-grounded claim verification and conflict-report analysis, with abstention.**

CivIntel-AgentBench evaluates how well systems verify claims about the Russia-Ukraine conflict from
evidence in English, Ukrainian, Russian and Chinese, when they should abstain, and whether they read
the evidence or shortcuts in it. It does not decide whether an event happened, attribute legal
responsibility, or replace professional fact-checking and investigation.

## Results on RU22Fact (v0.4.0)

Paper: [*Reading the Format, Not the Facts: Artefacts in LLM-Optimised Evidence Inflate Multilingual
Fact-Checking Scores*](paper/main.pdf).

Data: RU22Fact (Zeng et al., LREC-COLING 2024), 16,031 claims after dropping two malformed rows, with
the dataset's own splits (test: 1,741 English, 704 Ukrainian, 465 Russian and 305 Chinese claims).
Models: character n-gram TF-IDF with class-balanced logistic regression, regularisation chosen on
validation, test scored once. Macro-F1 over Supported, Refuted and NEI; per language over the labels
present (Ukrainian and Russian have no NEI claims); 95% bootstrap intervals over claims.

| Input | Test macro-F1 [95% CI] | Accuracy | en | uk | ru | zh | Accuracy at 50% coverage |
|---|---|---|---|---|---|---|---|
| Majority (Supported) | 0.268 | 0.674 | | | | | |
| claim | 0.614 [0.594, 0.633] | 0.779 | 0.584 | 0.868 | 0.950 | 0.572 | 0.905 |
| evidence | 0.677 [0.655, 0.696] | 0.858 | 0.631 | 0.989 | 0.995 | 0.696 | 0.980 |
| claim+evidence | 0.679 [0.659, 0.698] | 0.859 | 0.631 | 0.989 | 0.995 | 0.713 | 0.981 |
| evidence format only (no words) | 0.591 [0.574, 0.609] | 0.761 | 0.543 | 0.944 | 0.931 | 0.352 | |
| evidence, artefacts removed | 0.678 [0.657, 0.696] | | 0.630 | 0.989 | 0.971 | 0.710 | |
| format only, evidence, artefacts removed | 0.396 | | 0.117 | 0.944 | 0.493 | 0.330 | |
| evidence, artefacts removed, first 300 characters | 0.636 [0.618, 0.655] | | 0.581 | 0.970 | 0.931 | 0.685 | |
| format only, evidence, artefacts removed, first 300 characters | 0.357 | | 0.046 | 0.939 | 0.463 | 0.096 | |
| claim+evidence, trained on English only | | | 0.641 | 0.236 | 0.927 | 0.503 | |

Fine-tuned multilingual encoders on claim and evidence (Kaggle T4 GPU, three seeds each; 256 tokens, 3 epochs,
learning rate 2e-5, class-weighted loss, epoch chosen on validation macro-F1). Outputs: `results/ru22fact-neural/`.

| Model | Evidence | Test macro-F1, mean ± SD | en | uk | ru | zh |
|---|---|---|---|---|---|---|
| XLM-R base | raw | 0.689 ± 0.003 | 0.643 | 1.000 | 0.997 | 0.676 |
| XLM-R base | artefacts removed | 0.691 ± 0.005 | 0.644 | 0.999 | 0.997 | 0.700 |
| multilingual-E5 base | raw | 0.701 ± 0.008 | 0.652 | 0.999 | 0.999 | 0.709 |
| multilingual-E5 base | artefacts removed | 0.697 ± 0.008 | 0.647 | 0.999 | 0.997 | 0.710 |

**Findings.**

1. **The format predicts the label.** A classifier that reads no words of the evidence, only six
   format features (a search-engine greeting, "[1]" citation markers, list numbering, length,
   character shares), reaches macro-F1 0.591, which is 79% of the gain of the claim-and-evidence
   model (0.679) over the majority class (0.268). In Russian, 95% of Supported but 12% of Refuted
   evidence contains the search engine's greeting; in English, 83% of Supported and 13% of Refuted
   evidence starts with "[1]"; in Ukrainian the median Supported evidence is 107 characters long
   against 2,830 for Refuted.
2. **Removing the marks removes the format signal, not the score.** After artefact removal the
   format-only probe falls to 0.396, but the evidence model keeps 0.678.
3. **The evidence states the verdict.** 77.5% of Refuted Ukrainian evidence contains a debunking
   word, against 0.6% of Supported evidence (Russian 29.8% against 5.9%, English 28.6% against 5.2%).
   Scores of 0.99 on Ukrainian and Russian measure reading a verdict that is already in the
   evidence, not verifying a claim.
4. **Claims carry cues too** (claim alone: 0.614), **NEI is the hard class** (F1 0.262), and
   answering only the most confident half of the claims raises accuracy from 0.859 to 0.981.
5. **Lexical models do not transfer across scripts:** trained on English only, the model scores
   0.236 on Ukrainian; the higher Russian score (0.927) comes from shared format marks.
6. **Neural encoders show the same picture.** XLM-R and multilingual-E5 reach 0.689 and 0.701,
   keep their scores after artefact removal, and score 0.997-1.000 on Ukrainian and Russian.

Evaluations on LLM-built evidence should report a format-only baseline, scores after artefact
removal, and a verdict-word check. Next: tool-using agents that retrieve their own evidence, on the
same protocol.

Full numbers: [`results/ru22fact-v0.3.0/`](results/ru22fact-v0.3.0/).

## Quick start (macOS or Linux, Python 3.9+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
civintel ru22fact-download            # fetches RU22Fact from its pinned commit, checks SHA-256
civintel ru22fact-verify --output reports/ru22fact   # about 15 minutes on a laptop CPU
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
  verification.py   baselines, bootstrap intervals, risk-coverage, transfer, format probe,
                    artefact removal, verdict-word rates
  cli.py            civintel command
  pipeline.py, retrieval.py, extraction.py, agents.py, evaluation.py, io.py   v0.1.0 pipeline
results/ru22fact-v0.3.0/   results.json, results.md, run_manifest.json
paper/                     main.tex, main.pdf, references.bib, make_figure.py
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
