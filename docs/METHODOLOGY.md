# Methodology

## Task definition

Given a set of public report texts, produce candidate report pairs that may refer to a related incident and an auditable record of why they were surfaced. The unit of analysis is a **report**, not a person, organisation, or legal finding.

## Pipeline

1. Validate schema and unique report identifiers.
2. Extract transparent lexical features while preserving original text.
3. Build character n-gram TF-IDF representations.
4. Retrieve top-k candidate pairs using cosine similarity.
5. attach source and publication provenance;
6. mark possible cross-source matches while excluding same-source-family pairs from independent corroboration;
7. abstain from factual conclusions and emit a human-review warning.

## Evaluation design

Use a manually adjudicated pair-label file with both positive and hard-negative pairs. Split by event cluster and source family, not random individual rows, to reduce leakage. Report precision, recall, F1, retrieval recall@k, and per-language breakdowns. For confidence calibration in later neural/agentic versions, report Brier score, expected calibration error, risk-coverage curves, and abstention utility.

## Limitations

Character n-gram TF-IDF is not semantic multilingual understanding. Lexical matching may miss paraphrases and may match unrelated reports that share generic vocabulary. The v0.1.0 heuristic is a baseline for infrastructure validation, not a validated event linker.
