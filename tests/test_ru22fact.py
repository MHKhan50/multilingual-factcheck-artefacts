import numpy as np
import pandas as pd
import pytest

from civintel_agentbench.ru22fact import FILES, normalise, raw_url
from civintel_agentbench.verification import macro_f1, risk_coverage, run_verification


def toy(n_per=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    words = {"Supported": "confirmed verified official", "Refuted": "fake false hoax", "NEI": "unclear unknown rumour"}
    for split in ("train", "validate", "test"):
        for lang in ("English", "Ukrainian", "Russian", "Chinese"):
            for label, cue in words.items():
                for i in range(n_per):
                    rows.append({"split": split, "language": lang, "label": label,
                                 "claim": "claim {} {}".format(i, rng.integers(1000)),
                                 "evidence": "{} report {}".format(cue, i)})
    return normalise(pd.DataFrame(rows))


def test_pinned_download_urls_and_digests():
    assert set(FILES) == {"train", "validate", "test"}
    for name, digest in FILES.values():
        assert len(digest) == 64
        assert raw_url(name).startswith("https://raw.githubusercontent.com/zeng-yirong/ru22fact/c467abb")


def test_normalise_drops_malformed_rows_and_maps_languages():
    frame = pd.DataFrame([
        {"claim": "a", "evidence": "b", "label": "Supported", "language": "English"},
        {"claim": "c", "evidence": None, "label": "Refuted", "language": None},
    ])
    out = normalise(frame)
    assert list(out["lang"]) == ["en"]
    assert out.attrs["dropped_malformed"] == 1


def test_normalise_rejects_unknown_labels():
    with pytest.raises(ValueError):
        normalise(pd.DataFrame([{"claim": "a", "evidence": "b", "label": "Maybe", "language": "English"}]))


def test_risk_coverage_on_a_perfect_and_a_wrong_classifier():
    classes = np.array(["NEI", "Refuted", "Supported"])
    proba = np.array([[0.1, 0.1, 0.8], [0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])
    assert risk_coverage(["Supported", "Refuted", "NEI"], proba, classes)["accuracy_at_100%"] == 1.0
    assert risk_coverage(["NEI", "NEI", "Supported"], proba, classes)["accuracy_at_100%"] == 0.0


def test_evidence_cues_are_learned_and_claims_alone_are_not():
    results = run_verification(toy(), bootstrap=50)
    assert results["evidence"]["macro_f1"] == pytest.approx(1.0)
    assert results["claim"]["macro_f1"] < 0.7
    assert set(results["evidence"]["per_language"]) == {"en", "uk", "ru", "zh"}
    assert macro_f1(["Supported"], ["Supported"]) == pytest.approx(1 / 3)


def test_format_probe_reads_labels_from_format_artefacts():
    from civintel_agentbench.verification import format_features, run_format_probe

    data = toy()
    supported = data["label"] == "Supported"
    data.loc[supported, "evidence"] = "Hello, this is Bing. " + data.loc[supported, "evidence"]
    feats = format_features(data["evidence"])
    assert feats.loc[supported, "search_greeting"].eq(1.0).all()
    assert feats.loc[~supported, "search_greeting"].eq(0.0).all()
    probe = run_format_probe(data, bootstrap=20)
    assert probe["per_language"]["en"]["macro_f1"] > 0.4
    assert probe["feature_rates_by_label"]["en"]["search_greeting"]["Supported"] == 1.0


def test_per_language_macro_f1_uses_only_labels_present():
    from civintel_agentbench.verification import per_language

    test = pd.DataFrame({"lang": ["uk", "uk"], "label": ["Supported", "Refuted"]})
    out = per_language(test, ["Supported", "Refuted"])
    assert out["uk"]["labels"] == ["Supported", "Refuted"]
    assert out["uk"]["macro_f1"] == pytest.approx(1.0)
