"""RU22Fact: download, verification and loading.

RU22Fact (Zeng et al., LREC-COLING 2024) holds 16,033 fact-checked claims about the Russia-Ukraine
conflict in English, Ukrainian, Russian and Chinese, each with LLM-optimised evidence, a reference
explanation and a label (Supported, Refuted, NEI). The source repository carries no licence file, so
this project never redistributes the data: it downloads the three CSV files from a pinned commit and
checks their SHA-256 digests.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from .utils import sha256_file

REPOSITORY = "https://github.com/zeng-yirong/ru22fact"
COMMIT = "c467abb1c98bd4fc043ac0fd5a03c4f1afa109d0"
FILES = {
    "train": ("ru22fact_train.csv", "9dc0f12d5d3b1a4a88a437c5630f216a31196ffd32b3ffac5d815c41d4726db8"),
    "validate": ("ru22fact_validate.csv", "9b6e6b38b478204a28229791fde50385854b12309f2d60a49159c7f9f159f658"),
    "test": ("ru22fact_test.csv", "946b548f511deefac6898aad02189c809653e017aeda0f1f439def830a8a3aaf"),
}
LABELS = ("Supported", "Refuted", "NEI")
LANGUAGES = {"English": "en", "Ukrainian": "uk", "Russian": "ru", "Chinese": "zh"}


def raw_url(name: str) -> str:
    return "https://raw.githubusercontent.com/zeng-yirong/ru22fact/{}/data/ru22fact/{}".format(COMMIT, name)


def download(target_dir: str | Path, force: bool = False) -> dict[str, str]:
    """Download the three splits into ``target_dir`` and verify them; return the digests."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split, (name, expected) in FILES.items():
        path = target / name
        if force or not path.exists() or sha256_file(path) != expected:
            urllib.request.urlretrieve(raw_url(name), path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("Checksum mismatch for {}: {} != {}".format(name, actual, expected))
        digests[split] = actual
    return digests


def load(data_dir: str | Path, verify: bool = True) -> pd.DataFrame:
    """All splits in one frame with columns split, claim, evidence, label, language (ISO code)."""
    frames = []
    for split, (name, expected) in FILES.items():
        path = Path(data_dir) / name
        if verify and sha256_file(path) != expected:
            raise ValueError("Checksum mismatch for {}; run `civintel ru22fact-download` again".format(name))
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return normalise(data)


def normalise(data: pd.DataFrame) -> pd.DataFrame:
    """Check labels and languages and add ISO codes. Rows without a language are malformed in the
    source (their columns are shifted: one in train, one in test) and are dropped; the count is kept
    in ``attrs["dropped_malformed"]``."""
    malformed = data["language"].isna()
    data = data[~malformed]
    unknown_labels = set(data["label"]) - set(LABELS)
    if unknown_labels:
        raise ValueError("Unknown labels: {}".format(sorted(unknown_labels)))
    unknown_languages = set(data["language"]) - set(LANGUAGES)
    if unknown_languages:
        raise ValueError("Unknown languages: {}".format(sorted(unknown_languages)))
    out = data.copy()
    out["lang"] = out["language"].map(LANGUAGES)
    for column in ("claim", "evidence"):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out = out.reset_index(drop=True)
    out.attrs["dropped_malformed"] = int(malformed.sum())
    return out
