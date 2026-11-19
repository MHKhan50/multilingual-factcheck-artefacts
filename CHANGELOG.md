# Changelog

## Unreleased

- `civintel ru22fact-neural` and `notebooks/kaggle_xlmr.ipynb`: fine-tune `xlm-roberta-base` on claim
  and evidence pairs, three seeds, on raw and on artefact-removed evidence (Kaggle GPU).

## 0.3.0 (2026-11-19)

- Artefact control: evidence and format-only models refitted after removing search-engine sentences,
  citation markers, list numbering and markdown, and with the first 300 characters only.
- Verdict-word rates per language and label.
- Paper in `paper/` (LaTeX source, PDF, figure script); results of the full run in
  `results/ru22fact-v0.3.0/` (replacing `results/ru22fact-v0.2.0/`, whose numbers are unchanged).
- Zenodo DOI badge.

## 0.2.0 (2026-11-19)

- RU22Fact support: `civintel ru22fact-download` fetches the three splits from a pinned commit and
  checks SHA-256 digests (the data are not redistributed); two malformed source rows are dropped
  and counted.
- `civintel ru22fact-verify`: claim-verification baselines (majority; character n-gram TF-IDF with
  logistic regression on the claim, the evidence, or both), regularisation chosen on validation,
  test macro-F1 with 95% bootstrap intervals, per-language and per-class results, risk-coverage
  (abstention) metrics, English-only cross-lingual transfer, and a format-only probe of the
  evidence.
- Author metadata, `CITATION.cff`, continuous integration on Python 3.9 and 3.12; unused
  dependencies removed; corrected RU22Fact citation (LREC-COLING 2024, paper 1239) and SWARM
  description in `docs/DATA_SOURCES.md`.

## 0.1.0 (2026-11-19)

- Deterministic lexical pipeline for multilingual report matching and evidence audit on synthetic
  fixtures.
