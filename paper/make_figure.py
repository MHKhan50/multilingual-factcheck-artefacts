"""Figure 1: per-language macro-F1 from results/ru22fact-v0.3.0/results.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "results" / "ru22fact-v0.3.0" / "results.json").read_text())
langs = ["en", "uk", "ru", "zh"]
names = {"en": "English", "uk": "Ukrainian", "ru": "Russian", "zh": "Chinese"}
series = [
    ("Claim and evidence", [r["claim+evidence"]["per_language"][k]["macro_f1"] for k in langs], "#1F4E9C"),
    ("Format only (no words)", [r["format_only"]["per_language"][k]["macro_f1"] for k in langs], "#E3A800"),
    ("Format only, artefacts removed", [r["artefact_control"]["cleaned"]["format_only_per_language"][k]["macro_f1"] for k in langs], "#C8102E"),
]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
width = 0.26
for i, (label, values, color) in enumerate(series):
    xs = [j + (i - 1) * width for j in range(len(langs))]
    ax.bar(xs, values, width - 0.02, label=label, color=color)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.015, "{:.2f}".format(v), ha="center", fontsize=7)
ax.set_xticks(range(len(langs)))
ax.set_xticklabels([names[k] for k in langs])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Test macro-F1")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout()
out = ROOT / "paper" / "figures" / "fig_format_probe.pdf"
fig.savefig(out)
print("wrote", out)
