"""Claim-verification baselines on RU22Fact with per-language results and abstention.

Inputs are the claim, the LLM-optimised evidence, or both. The reference explanation is never used
as model input, because it states the fact-checkers' verdict. Character n-gram TF-IDF with logistic
regression gives transparent, CPU-only baselines; the regularisation strength is chosen on the
validation split and the test split is scored once per setting.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from .ru22fact import LABELS

INPUTS = {
    "claim": lambda d: d["claim"],
    "evidence": lambda d: d["evidence"],
    "claim+evidence": lambda d: d["claim"] + " [SEP] " + d["evidence"],
}
C_GRID = (0.3, 1.0, 3.0)
COVERAGES = (0.5, 0.8, 1.0)


def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0) -> list[float]:
    """95% percentile interval of macro-F1 over test claims resampled with replacement."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(macro_f1(y_true[idx], y_pred[idx]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def make_model(c: float, seed: int):
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=seed),
    )


def fit_select(train: pd.DataFrame, val: pd.DataFrame, text, seed: int):
    """Fit one model per C on train, keep the best on validation macro-F1."""
    best = None
    for c in C_GRID:
        model = make_model(c, seed).fit(text(train), train["label"])
        score = macro_f1(val["label"], model.predict(text(val)))
        if best is None or score > best[0]:
            best = (score, c, model)
    return best


def risk_coverage(y_true, proba: np.ndarray, classes) -> dict[str, float]:
    """Accuracy when only the most confident share of claims is answered; AURC over all coverages."""
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    order = np.argsort(-proba.max(axis=1), kind="stable")
    correct = (pred[order] == y_true[order]).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    out = {"aurc": float(np.mean(1.0 - cum_acc))}
    for cov in COVERAGES:
        k = max(1, int(round(cov * len(correct))))
        out["accuracy_at_{:.0%}".format(cov)] = float(cum_acc[k - 1])
    return out


def per_language(test: pd.DataFrame, pred) -> dict[str, dict[str, float]]:
    out = {}
    pred = pd.Series(np.asarray(pred), index=test.index)
    for lang, group in test.groupby("lang"):
        # Ukrainian and Russian have no NEI claims; macro-F1 is taken over the labels present.
        present = [label for label in LABELS if (group["label"] == label).any()]
        out[lang] = {
            "n": int(len(group)),
            "labels": present,
            "macro_f1": macro_f1(group["label"], pred[group.index], labels=present),
            "accuracy": float(accuracy_score(group["label"], pred[group.index])),
        }
    return out


def run_verification(data: pd.DataFrame, seed: int = 0, bootstrap: int = 1000) -> dict:
    train = data[data["split"] == "train"]
    val = data[data["split"] == "validate"]
    test = data[data["split"] == "test"]
    results: dict = {"n": {s: int((data["split"] == s).sum()) for s in ("train", "validate", "test")}}

    majority = train["label"].value_counts().idxmax()
    maj_pred = [majority] * len(test)
    results["majority"] = {
        "label": majority,
        "macro_f1": macro_f1(test["label"], maj_pred),
        "accuracy": float(accuracy_score(test["label"], maj_pred)),
    }

    for name, text in INPUTS.items():
        val_score, c, model = fit_select(train, val, text, seed)
        proba = model.predict_proba(text(test))
        classes = model.classes_
        pred = np.asarray(classes)[proba.argmax(axis=1)]
        results[name] = {
            "C": c,
            "val_macro_f1": val_score,
            "macro_f1": macro_f1(test["label"], pred),
            "macro_f1_ci95": bootstrap_ci(test["label"], pred, n=bootstrap, seed=seed),
            "accuracy": float(accuracy_score(test["label"], pred)),
            "per_class_f1": dict(zip(LABELS, f1_score(test["label"], pred, labels=list(LABELS), average=None, zero_division=0).round(4).tolist())),
            "per_language": per_language(test, pred),
            "selective": risk_coverage(test["label"], proba, classes),
        }

    # Cross-lingual transfer: train on English only, test on every language.
    text = INPUTS["claim+evidence"]
    en_train = train[train["lang"] == "en"]
    en_val = val[val["lang"] == "en"]
    _, c, model = fit_select(en_train, en_val, text, seed)
    pred = model.predict(text(test))
    results["english_only_claim+evidence"] = {"C": c, "per_language": per_language(test, pred)}
    results["format_only"] = run_format_probe(data, seed=seed, bootstrap=bootstrap)
    results["artefact_control"] = run_artefact_control(data, seed=seed, bootstrap=bootstrap)
    results["verdict_cues"] = verdict_cue_rates(data)
    return results


def results_table(results: dict) -> str:
    rows = [
        "| Input | Test macro-F1 [95% CI] | Accuracy | en | uk | ru | zh | Accuracy at 50% coverage |",
        "|---|---|---|---|---|---|---|---|",
        "| Majority ({}) | {:.3f} | {:.3f} | | | | | |".format(
            results["majority"]["label"], results["majority"]["macro_f1"], results["majority"]["accuracy"]),
    ]
    for name in INPUTS:
        r = results[name]
        lang = r["per_language"]
        rows.append("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {} | {:.3f} |".format(
            name, r["macro_f1"], r["macro_f1_ci95"][0], r["macro_f1_ci95"][1], r["accuracy"],
            " | ".join("{:.3f}".format(lang[k]["macro_f1"]) for k in ("en", "uk", "ru", "zh")),
            r["selective"]["accuracy_at_50%"]))
    f = results["format_only"]
    rows.append("| evidence format only (no words) | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {} | |".format(
        f["macro_f1"], f["macro_f1_ci95"][0], f["macro_f1_ci95"][1], f["accuracy"],
        " | ".join("{:.3f}".format(f["per_language"][k]["macro_f1"]) for k in ("en", "uk", "ru", "zh"))))
    for name, label in (("cleaned", "evidence, artefacts removed"), ("cleaned_first_300", "evidence, artefacts removed, first 300 characters")):
        a = results["artefact_control"][name]
        rows.append("| {} | {:.3f} [{:.3f}, {:.3f}] | | {} | |".format(
            label, a["evidence_macro_f1"], a["evidence_macro_f1_ci95"][0], a["evidence_macro_f1_ci95"][1],
            " | ".join("{:.3f}".format(a["evidence_per_language"][k]["macro_f1"]) for k in ("en", "uk", "ru", "zh"))))
        rows.append("| format only, {} | {:.3f} | | {} | |".format(
            label, a["format_only_macro_f1"],
            " | ".join("{:.3f}".format(a["format_only_per_language"][k]["macro_f1"]) for k in ("en", "uk", "ru", "zh"))))
    x = results["english_only_claim+evidence"]["per_language"]
    rows.append("| claim+evidence, trained on English only | | | {} | |".format(
        " | ".join("{:.3f}".format(x[k]["macro_f1"]) for k in ("en", "uk", "ru", "zh"))))
    return "\n".join(rows)


def format_features(evidence: pd.Series) -> pd.DataFrame:
    """Features of how the evidence is formatted, not what it says.

    The evidence was produced by an LLM with web search; artefacts of that process (a search-engine
    greeting, "[1]:" citation markers, numbered lists, length) are content-free. A classifier on these
    features alone measures how much of the label can be read from the format.
    """
    text = evidence.fillna("").astype(str)
    length = text.str.len().clip(lower=1)
    return pd.DataFrame({
        "search_greeting": text.str.contains("Bing", regex=False).astype(float),
        "bracket_citations": text.str.match(r"\s*\[1\]").astype(float),
        "numbered_list": text.str.match(r"\s*1\.").astype(float),
        "log_length": np.log(length),
        "latin_share": text.str.count(r"[A-Za-z]") / length,
        "digit_share": text.str.count(r"\d") / length,
    })


def run_format_probe(data: pd.DataFrame, seed: int = 0, bootstrap: int = 1000) -> dict:
    """Format-only baseline, one model over all languages with a language indicator."""
    def features(frame: pd.DataFrame) -> pd.DataFrame:
        f = format_features(frame["evidence"])
        for lang in ("en", "uk", "ru", "zh"):
            f["lang_" + lang] = (frame["lang"] == lang).astype(float)
            for col in ("search_greeting", "bracket_citations", "numbered_list", "log_length"):
                f["{}_x_{}".format(col, lang)] = f[col] * f["lang_" + lang]
        return f

    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    model = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
    model.fit(features(train), train["label"])
    pred = model.predict(features(test))
    rates = {}
    for lang, group in data.groupby("lang"):
        f = format_features(group["evidence"]).assign(label=group["label"].values)
        rates[lang] = f.groupby("label")[["search_greeting", "bracket_citations"]].mean().round(3).to_dict()
        rates[lang]["median_length"] = group.groupby("label")["evidence"].apply(lambda s: int(s.str.len().median())).to_dict()
    return {
        "macro_f1": macro_f1(test["label"], pred),
        "macro_f1_ci95": bootstrap_ci(test["label"], pred, n=bootstrap, seed=seed),
        "accuracy": float(accuracy_score(test["label"], pred)),
        "per_language": per_language(test, pred),
        "feature_rates_by_label": rates,
    }


_CITATION = re.compile(r"\[\^?\d+\^?\]:?")
_NUMBERING = re.compile(r"(^|\n)\s*\d+[.)]\s+")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+")


def clean_evidence(text: str, max_chars: int | None = None) -> str:
    """Remove formatting artefacts: sentences that mention the search engine, citation markers,
    list numbering and markdown emphasis; optionally keep only the first ``max_chars`` characters."""
    text = str(text).replace("**", " ")
    text = _CITATION.sub(" ", text)
    text = _NUMBERING.sub(" ", text)
    text = " ".join(s for s in _SENTENCE.split(text) if "Bing" not in s)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if max_chars else text


ARTEFACT_VARIANTS = {"raw": None, "cleaned": 0, "cleaned_first_300": 300}


def run_artefact_control(data: pd.DataFrame, seed: int = 0, bootstrap: int = 1000) -> dict:
    """Evidence-only and format-only models on raw evidence and on evidence with artefacts removed.

    C is fixed at 1.0 (the value selected on validation for the raw inputs) so that the variants
    differ only in the evidence text.
    """
    out = {}
    for name, cap in ARTEFACT_VARIANTS.items():
        frame = data.copy()
        if cap is not None:
            frame["evidence"] = frame["evidence"].map(lambda t: clean_evidence(t, cap or None))
        train = frame[frame["split"] == "train"]
        test = frame[frame["split"] == "test"]
        model = make_model(1.0, seed).fit(train["evidence"], train["label"])
        pred = model.predict(test["evidence"])
        probe = run_format_probe(frame, seed=seed, bootstrap=bootstrap)
        out[name] = {
            "median_chars": int(test["evidence"].str.len().median()),
            "evidence_macro_f1": macro_f1(test["label"], pred),
            "evidence_macro_f1_ci95": bootstrap_ci(test["label"], pred, n=bootstrap, seed=seed),
            "evidence_per_language": per_language(test, pred),
            "format_only_macro_f1": probe["macro_f1"],
            "format_only_per_language": probe["per_language"],
        }
    return out


# Words with which fact-checkers state a verdict, per language (a small, transparent list).
VERDICT_CUES = {
    "en": r"\b(?:false|misleading|fake|no evidence|debunk|fact.check)",
    "uk": r"(?:фейк|неправд|не відповідає|спростов|маніпуляц)",
    "ru": r"(?:фейк|неправд|ложн|не соответству|опроверг|манипуляц)",
    "zh": r"(?:假|不实|谣言|误导|错误)",
}


def mask_verdict_words(text: str, lang: str) -> str:
    """Delete every word that contains a verdict cue of the claim's language (Chinese: the cue itself)."""
    pattern = VERDICT_CUES.get(lang)
    if pattern is None:
        return str(text)
    core = pattern.replace(r"\b", "")
    if lang != "zh":
        core = r"\w*" + core + r"\w*"
    return re.sub(r"\s+", " ", re.sub(core, " ", str(text), flags=re.IGNORECASE)).strip()


def masked_evidence(data: pd.DataFrame) -> pd.Series:
    """Artefact-removed evidence with the verdict words of each claim's language deleted."""
    return pd.Series([mask_verdict_words(clean_evidence(t), lang) for t, lang in zip(data["evidence"], data["lang"])],
                     index=data.index)


def run_masked_control(data: pd.DataFrame, seed: int = 0, bootstrap: int = 1000) -> dict:
    """Lexical models on artefact-removed evidence with and without verdict words (C fixed at 1.0)."""
    out = {}
    for name in ("cleaned", "cleaned_masked"):
        frame = data.copy()
        frame["evidence"] = masked_evidence(frame) if name == "cleaned_masked" else frame["evidence"].map(clean_evidence)
        train = frame[frame["split"] == "train"]
        test = frame[frame["split"] == "test"]
        out[name] = {}
        for key in ("evidence", "claim+evidence"):
            text = INPUTS[key]
            pred = make_model(1.0, seed).fit(text(train), train["label"]).predict(text(test))
            out[name][key] = {"macro_f1": macro_f1(test["label"], pred),
                              "macro_f1_ci95": bootstrap_ci(test["label"], pred, n=bootstrap, seed=seed),
                              "per_language": per_language(test, pred)}
        out[name]["verdict_cue_rates"] = verdict_cue_rates(frame[frame["split"] == "test"])
    return out


def verdict_cue_rates(data: pd.DataFrame) -> dict:
    """Share of evidence texts per language and label that contain a verdict word."""
    out = {}
    for lang, pattern in VERDICT_CUES.items():
        group = data[data["lang"] == lang]
        hits = group["evidence"].str.contains(pattern, case=False, regex=True)
        out[lang] = {label: round(float(hits[group["label"] == label].mean()), 3)
                     for label in LABELS if (group["label"] == label).any()}
    return out
