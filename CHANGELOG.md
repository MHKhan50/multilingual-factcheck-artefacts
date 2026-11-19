# Changelog

## 0.2.0 (2026-11-19)

- RU22Fact support: `civintel ru22fact-download` fetches the three splits from a pinned commit and
  checks SHA-256 digests (the data are not redistributed); two malformed source rows are dropped
  and counted.
- `civintel ru22fact-verify`: claim-verification baselines (majority; character n-gram TF-IDF with
  logistic regression on the claim, the evidence, or both), regularisation chosen on validation,
  test macro-F1 with 95% bootstrap intervals, per-language and per-class results, risk-coverage
  (abstention) metrics, English-only cross-lingual transfer, and a format-only probe of the
  evidence.
- Results of the first full run archived in `results/ru22fact-v0.2.0/`.
- Author metadata, `CITATION.cff`, continuous integration on Python 3.9 and 3.12; unused
  dependencies removed; corrected RU22Fact citation (LREC-COLING 2024, paper 1239) and SWARM
  description in `docs/DATA_SOURCES.md`.

## 0.1.0 (2026-11-19)

- Deterministic lexical pipeline for multilingual report matching and evidence audit on synthetic
  fixtures.
