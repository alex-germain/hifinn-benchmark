"""
Figures for the HiFi-NN benchmark on swissprot_July31_cutoff.

Reads evaluation_results.json (written by evaluate.py) and produces two PNGs:
  fig1_performance_by_ec_level.png  — accuracy and macro-F1 per EC level, with
                                      bootstrap 95% CIs and two null baselines
  fig2_label_conventions.png        — effect of the convention used for partial
                                      ground-truth annotations

Figure text is in English: the repository is public and English-language.

Usage: python scripts/make_figures.py
"""

import json
import os

try:
    import matplotlib
except ImportError:
    raise SystemExit(
        "matplotlib est requis pour les figures : pip install matplotlib\n"
        "Les resultats chiffres sont produits par scripts/evaluate.py."
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS as RESULTS_DIR, FIGURES, need

RESULTS = os.path.join(RESULTS_DIR, "evaluation_results.json")
FIGDIR = FIGURES

# palette validee (ecarts CVD et vision normale verifies sur les trois teintes)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

LEVELS = [1, 2, 3, 4]
LEVEL_NAMES = ["1\nclass", "2\nsubclass", "3\nsub-subclass", "4\nfull EC"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "savefig.facecolor": SURFACE,
})


def style(ax, ylabel):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(LEVELS)
    ax.set_xticklabels(LEVEL_NAMES, fontsize=8.5)
    ax.set_xlim(0.75, 4.95)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold",
                va="center", ha="left")


def fig1(res):
    conv = {l: res["levels"][str(l)]["conventions"]["B"] for l in LEVELS}
    base = {l: res["levels"][str(l)]["baselines"] for l in LEVELS}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax, key, ci_key, title in (
        (axes[0], "accuracy", "accuracy_ci95", "Top-1 accuracy"),
        (axes[1], "macro_f1", "macro_f1_ci95", "Macro-F1"),
    ):
        y = [conv[l][key] for l in LEVELS]
        ci = [conv[l][ci_key] for l in LEVELS]
        lo = [v - c[0] for v, c in zip(y, ci)]
        hi = [c[1] - v for v, c in zip(y, ci)]

        ax.errorbar(LEVELS, y, yerr=[lo, hi], color=BLUE, linewidth=2,
                    marker="o", markersize=8, capsize=4, elinewidth=1.4,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                    label="HiFi-NN")
        ym = [base[l]["majority"][key] for l in LEVELS]
        yr = [base[l]["weighted_random"][key] for l in LEVELS]
        ax.plot(LEVELS, ym, color=ORANGE, linewidth=2, marker="s", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2,
                label="Majority class")
        ax.plot(LEVELS, yr, color=AQUA, linewidth=2, marker="^", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2,
                label="Weighted random")

        label_end(ax, 4, y[-1], f"{y[-1]:.2f}", BLUE)
        label_end(ax, 4, ym[-1], f"{ym[-1]:.2f}", ORANGE, dy=9)
        label_end(ax, 4, yr[-1], f"{yr[-1]:.2f}", AQUA, dy=-9)

        style(ax, title)
        ax.set_xlabel("EC hierarchy level", fontsize=9.5, color=INK2)

    axes[0].legend(frameon=False, loc="upper right", fontsize=9,
                   labelcolor=INK2, handlelength=1.6,
                   bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("HiFi-NN degrades with EC depth but stays far above chance",
                 fontsize=12, fontweight="bold", color=INK, x=0.5, y=0.99)
    fig.text(0.5, 0.015,
             "Swiss-Prot July-31 cutoff, 244 sequences, none present in the shipped FAISS index. "
             "Masked convention. Error bars: bootstrap 95% CI, 2,000 resamples.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    out = os.path.join(FIGDIR, "fig1_performance_by_ec_level.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig2(res):
    convs = [("A", "A — strict", ORANGE, "s"),
             ("B", "B — masked", BLUE, "o"),
             ("C", "C — lenient", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for key, name, color, marker in convs:
        y = [res["levels"][str(l)]["conventions"][key]["accuracy"] for l in LEVELS]
        lw = 2.4 if key == "B" else 2
        ax.plot(LEVELS, y, color=color, linewidth=lw, marker=marker,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3 if key == "B" else 2, label=name)
        dy = {"A": -15, "B": 4, "C": 11}[key]
        label_end(ax, 4, y[-1], f"{y[-1]:.3f}", color, dy=dy)

    style(ax, "Top-1 accuracy")
    ax.set_xlabel("EC hierarchy level", fontsize=9.5, color=INK2)
    ax.legend(frameon=False, loc="lower left", fontsize=9,
              labelcolor=INK2, handlelength=1.6, bbox_to_anchor=(0.0, 0.08))

    ns = [res["levels"][str(l)]["conventions"]["B"]["n"] for l in LEVELS]
    for l, n in zip(LEVELS, ns):
        ax.annotate(f"n={n}", xy=(l, 0.035), ha="center", fontsize=8, color=MUTED)

    ax.set_title("The gap between conventions measures annotation incompleteness,\nnot model performance",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=12)
    fig.text(0.01, 0.015,
             "n = proteins evaluable under the masked convention. "
             "39% of the set has no level-4 annotation.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    out = os.path.join(FIGDIR, "fig2_label_conventions.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    res = json.load(open(need(RESULTS, "Lancer d'abord scripts/evaluate.py.")))
    for out in (fig1(res), fig2(res)):
        print("wrote:", out)


if __name__ == "__main__":
    main()
