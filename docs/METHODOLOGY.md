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

## v0.2.0: claim verification on RU22Fact

**Task.** Given a claim and the LLM-optimised evidence released with RU22Fact, predict Supported,
Refuted or NEI. The reference explanation is never a model input, because it states the verdict.

**Splits and selection.** The dataset's own train, validation and test splits are used. For each
input (claim, evidence, claim and evidence) a character n-gram (2-5) TF-IDF and class-balanced
logistic-regression model is fitted with C in {0.3, 1, 3}; the value with the best validation
macro-F1 is kept and the test split is scored once.

**Metrics.** Macro-F1 over the three labels (the main metric, because NEI is rare), accuracy,
per-class F1, per-language macro-F1, and a 95% percentile bootstrap interval over test claims
(1,000 resamples). For abstention, claims are ranked by the top predicted probability and the
accuracy of the most confident 50%, 80% and 100% is reported, with the area under the
risk-coverage curve (AURC, lower is better).

**Transfer.** The same model trained on English claims only is scored on every language.

**Format probe.** A logistic regression on content-free features of the evidence (a search-engine
greeting, "[1]" citation markers, a numbered list, length, the share of Latin letters and digits),
with language indicators, measures how much of the label can be predicted from how the evidence
is formatted rather than what it says.

**Interpretation.** Labels are the fact-checkers' verdicts as released; they are not
independently verified here, and no model output is a statement about the events.
