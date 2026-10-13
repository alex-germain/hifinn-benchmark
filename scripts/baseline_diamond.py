"""
DIAMOND best-hit transfer baseline, compared to HiFi-NN.

The question a benchmark of an embedding-based annotator has to answer is not
"how accurate is it" but "is it better than alignment". The classical method is
to take the best DIAMOND hit and transfer its EC numbers to the query.

Two comparisons are reported, because they say different things:

  aligned  — restricted to the 207 queries DIAMOND aligns at all. Measures
             accuracy on equal footing.
  all      — all queries; a query DIAMOND cannot align counts as an error.
             Measures accuracy and coverage together, which is what a user
             of an annotation tool actually experiences.

A third slice, the 37 queries with no DIAMOND alignment, isolates what the
embedding model adds where alignment returns nothing.

Input: diamond_hits.tsv, july31_annos.json,
       ../Datasets/swissprot_July31_cutoff_id_to_ec.json,
       ../ModelData/swissprot_all_ec_nums.json

Output: baseline_results.json, figures/fig4_vs_diamond.png

Usage: python scripts/baseline_diamond.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import depth, truncate, top1, score, build_pairs, bootstrap_ci

from paths import DATASETS, MODELDATA, RESULTS, FIGURES, need

HITS = os.path.join(RESULTS, "diamond_hits.tsv")
PRED = os.path.join(RESULTS, "july31_annos.json")
TRUE = os.path.join(DATASETS, "swissprot_July31_cutoff_id_to_ec.json")
INDEX_EC = os.path.join(MODELDATA, "swissprot_all_ec_nums.json")
OUT = os.path.join(RESULTS, "baseline_results.json")
FIGDIR = FIGURES

LEVELS = [1, 2, 3, 4]
SEED = 0

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def diamond_predictions(index_ec):
    """EC du meilleur hit (par bitscore) transferes a la requete."""
    best = {}
    with open(HITS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            q, s, bits = p[0], p[1], float(p[7])
            if q not in best or bits > best[q][1]:
                best[q] = (s, bits)
    pred = {}
    for q, (s, _) in best.items():
        ecs = index_ec.get(s)
        if ecs:
            pred[q] = {e: 1.0 for e in ecs}
    return pred, set(best)


def score_all(pred, true, level):
    """
    Comme build_pairs/score en convention B, mais une requete sans prediction
    compte comme une erreur au lieu d'etre ecartee.
    """
    pairs, n_missing = [], 0
    for pid, truths in true.items():
        d = max(depth(e) for e in truths)
        if d < level:
            continue
        t = {truncate(e, level) for e in truths if depth(e) >= level}
        if pid in pred and pred[pid]:
            pairs.append((t, truncate(top1(pred[pid]), level)))
        else:
            pairs.append((t, "__no_prediction__"))
            n_missing += 1
    s = score(pairs)
    s["n_missing"] = n_missing
    return s, pairs


def main():
    pred_h = json.load(open(need(PRED)))
    true = json.load(open(need(TRUE)))
    index_ec = json.load(open(need(INDEX_EC)))
    need(HITS)
    pred_d, aligned = diamond_predictions(index_ec)
    rng = random.Random(SEED)

    no_align = {k: v for k, v in true.items() if k not in aligned}
    only_align = {k: v for k, v in true.items() if k in pred_d}

    res = {
        "n_queries": len(true),
        "n_aligned": len(aligned),
        "n_transferable": len(pred_d),
        "n_no_alignment": len(no_align),
        "aligned": {}, "all": {}, "no_alignment": {},
    }

    for level in LEVELS:
        # 1. a armes egales, sur les requetes que DIAMOND aligne
        ph = build_pairs(pred_h, only_align, level, "B")
        pd_ = build_pairs(pred_d, only_align, level, "B")
        sh, sd = score(ph), score(pd_)
        for s, p in ((sh, ph), (sd, pd_)):
            if s["n"] >= 2:
                s["accuracy_ci95"] = bootstrap_ci(p, rng, n_boot=1000)[0]
        res["aligned"][str(level)] = {"hifinn": sh, "diamond": sd}

        # 2. sur tout le jeu, une absence de prediction = une erreur
        sha, pha = score_all(pred_h, true, level)
        sda, pda = score_all(pred_d, true, level)
        for s, p in ((sha, pha), (sda, pda)):
            if s["n"] >= 2:
                s["accuracy_ci95"] = bootstrap_ci(p, rng, n_boot=1000)[0]
        res["all"][str(level)] = {"hifinn": sha, "diamond": sda}

        # 3. la ou DIAMOND ne renvoie rien
        res["no_alignment"][str(level)] = {
            "hifinn": score(build_pairs(pred_h, no_align, level, "B")),
            "diamond": {"accuracy": 0.0, "note": "no alignment, no prediction"},
        }

    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\n{len(true)} requetes | DIAMOND en aligne {len(aligned)} "
          f"| {len(no_align)} sans aucun alignement\n")
    for name, title in (("aligned", "A armes egales (requetes alignees par DIAMOND)"),
                        ("all", "Sur tout le jeu (pas d'alignement = erreur)"),
                        ("no_alignment", "La ou DIAMOND ne renvoie rien")):
        print(title)
        print(f"  {'niv':>4} {'n':>5} {'DIAMOND':>9} {'HiFi-NN':>9}")
        for level in LEVELS:
            e = res[name][str(level)]
            h, d = e["hifinn"], e["diamond"]
            da = f"{d['accuracy']:.3f}" if d.get("accuracy") is not None else "   -  "
            print(f"  {level:>4} {h['n']:>5} {da:>9} {h['accuracy']:>9.3f}")
        print()

    figure(res)


def figure(res):
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(
            "matplotlib est requis pour les figures : pip install matplotlib\n"
            "Les resultats chiffres ont bien ete ecrits dans results/."
        )
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
        "savefig.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    x = np.arange(len(LEVELS))
    w = 0.36

    panels = [
        (axes[0], "aligned",
         f"Where DIAMOND aligns  (n={res['n_transferable']} queries)"),
        (axes[1], "all",
         f"All queries  (n={res['n_queries']}; no alignment counts as an error)"),
    ]

    for ax, key, title in panels:
        for off, who, color in ((-w / 2 - 0.01, "diamond", ORANGE),
                                (w / 2 + 0.01, "hifinn", BLUE)):
            y, lo, hi = [], [], []
            for level in LEVELS:
                s = res[key][str(level)][who]
                a = s["accuracy"]
                ci = s.get("accuracy_ci95") or (a, a)
                y.append(a)
                lo.append(a - ci[0])
                hi.append(ci[1] - a)
            bars = ax.bar(x + off, y, w, color=color,
                          label="DIAMOND best-hit" if who == "diamond" else "HiFi-NN",
                          zorder=2)
            for b in bars:
                b.set_linewidth(1.0)
                b.set_edgecolor(SURFACE)
            ax.errorbar(x + off, y, yerr=[lo, hi], fmt="none", ecolor=INK2,
                        elinewidth=1.1, capsize=3, zorder=3, alpha=0.75)
            for xi, v in zip(x + off, y):
                ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 12),
                            textcoords="offset points", ha="center",
                            fontsize=8.5, color=color, fontweight="bold")

        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticks(x)
        ax.set_xticklabels(["1\nclass", "2\nsubclass", "3\nsub-subclass",
                            "4\nfull EC"], fontsize=8.5)
        ax.set_xlabel("EC hierarchy level", fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=10, color=INK2, loc="left", pad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Top-1 accuracy", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=9,
               labelcolor=INK2, loc="upper right", bbox_to_anchor=(0.995, 1.0))

    fig.suptitle("HiFi-NN matches alignment where alignment works; its only gain is coverage",
                 fontsize=12, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.015,
             "Swiss-Prot July-31 cutoff, masked convention, bootstrap 95% CI (1,000 resamples). "
             "DIAMOND baseline: EC numbers of the best hit by bitscore,\ntransferred to the query. "
             "On the 37 queries DIAMOND cannot align at all, HiFi-NN reaches 0.68 at level 1 "
             "and 0.00 at level 4.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 0.925])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig4_vs_diamond.png")
    fig.savefig(out, dpi=200)
    print("wrote:", out)


if __name__ == "__main__":
    main()
