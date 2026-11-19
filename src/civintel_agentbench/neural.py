"""Fine-tuned multilingual transformer (XLM-R) on RU22Fact, on raw and on artefact-removed evidence.

The model reads the claim and the evidence as a sentence pair. One model is trained per seed and
evidence variant; the epoch with the best validation macro-F1 is kept and the test split is scored
once. Requires ``torch`` and ``transformers`` (preinstalled on Kaggle GPU images).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ru22fact import LABELS
from .verification import bootstrap_ci, clean_evidence, macro_f1, per_language, risk_coverage

VARIANTS = ("raw", "cleaned")


def prepare(data: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError("Unknown variant: {}".format(variant))
    out = data.copy()
    if variant == "cleaned":
        out["evidence"] = out["evidence"].map(clean_evidence)
    return out


def class_weights(labels: pd.Series) -> np.ndarray:
    """Inverse-frequency weights in LABELS order, normalised to mean 1."""
    counts = np.array([(labels == label).sum() for label in LABELS], dtype=float)
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))
    return weights / weights.mean()


def summarise(runs: list[dict]) -> dict:
    """Mean and standard deviation over seeds for the overall and per-language test macro-F1."""
    out: dict = {}
    for variant in sorted({r["variant"] for r in runs}):
        rs = [r for r in runs if r["variant"] == variant]
        f1 = [r["test"]["macro_f1"] for r in rs]
        langs = {}
        for lang in rs[0]["test"]["per_language"]:
            vals = [r["test"]["per_language"][lang]["macro_f1"] for r in rs]
            langs[lang] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        out[variant] = {
            "seeds": [r["seed"] for r in rs],
            "macro_f1": {"mean": float(np.mean(f1)), "std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0},
            "per_language": langs,
            "minutes": float(np.mean([r["minutes"] for r in rs])),
        }
    return out


def train_one(data: pd.DataFrame, variant: str, seed: int, model_name: str = "xlm-roberta-base",
              max_len: int = 256, epochs: int = 3, batch: int = 32, lr: float = 2e-5,
              limit: int | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = prepare(data, variant)
    splits = {s: frame[frame["split"] == s] for s in ("train", "validate", "test")}
    if limit:
        splits = {s: d.groupby("label", group_keys=False).head(limit) for s, d in splits.items()}
    tok = AutoTokenizer.from_pretrained(model_name)
    index = {label: i for i, label in enumerate(LABELS)}

    def encode(d: pd.DataFrame, shuffle: bool):
        enc = tok(list(d["claim"]), list(d["evidence"]), truncation="only_second", max_length=max_len,
                  padding="max_length", return_tensors="pt")
        y = torch.tensor([index[label] for label in d["label"]])
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=gen if shuffle else None)

    loaders = {"train": encode(splits["train"], True), "validate": encode(splits["validate"], False),
               "test": encode(splits["test"], False)}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS)).to(device)
    weights = torch.tensor(class_weights(splits["train"]["label"]), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(loaders["train"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def predict(loader) -> np.ndarray:
        model.eval()
        probs = []
        with torch.no_grad():
            for ids, mask, _ in loader:
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(probs)

    start = time.time()
    best = (-1.0, None, None)
    history = []
    for epoch in range(epochs):
        model.train()
        for ids, mask, y in loaders["train"]:
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        val_probs = predict(loaders["validate"])
        val_f1 = macro_f1(splits["validate"]["label"], np.array(LABELS)[val_probs.argmax(1)])
        history.append({"epoch": epoch + 1, "val_macro_f1": val_f1})
        if val_f1 > best[0]:
            best = (val_f1, epoch + 1, predict(loaders["test"]))
    test = splits["test"]
    probs = best[2]
    pred = np.array(LABELS)[probs.argmax(1)]
    return {
        "variant": variant, "seed": seed, "model": model_name, "max_len": max_len, "epochs": epochs,
        "best_epoch": best[1], "val_macro_f1": best[0], "history": history,
        "minutes": round((time.time() - start) / 60, 1),
        "test": {
            "macro_f1": macro_f1(test["label"], pred),
            "macro_f1_ci95": bootstrap_ci(test["label"], pred, n=1000, seed=seed),
            "accuracy": float((pred == test["label"].to_numpy()).mean()),
            "per_language": per_language(test, pred),
            "selective": risk_coverage(test["label"], probs, np.array(LABELS)),
        },
    }


def run_neural(data: pd.DataFrame, output: str | Path, seeds=(0, 1, 2), variants=VARIANTS, **kwargs) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in variants:
        for seed in seeds:
            result = train_one(data, variant, seed, **kwargs)
            runs.append(result)
            print(json.dumps({k: result[k] for k in ("variant", "seed", "best_epoch", "minutes")}
                             | {"test_macro_f1": round(result["test"]["macro_f1"], 4)}), flush=True)
            (out / "runs.json").write_text(json.dumps(runs, indent=2))
    summary = summarise(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
