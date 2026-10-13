"""
Identity-stratified evaluation of HiFi-NN on swissprot_July31_cutoff.

Average accuracy over a test set hides the only question that matters for an
annotation tool: does it still work when the query has no close homologue?
This script bins each query by its maximum sequence identity to the indexed
corpus and re-scores within each bin.

Input (all beside this script unless noted):
  diamond_hits.tsv   DIAMOND blastp, queries vs reduced30_train.fasta, tabular:
                     qseqid sseqid pident length qlen slen evalue bitscore
  july31_annos.json  HiFi-NN predictions
  ../Datasets/swissprot_July31_cutoff_id_to_ec.json   ground truth

The DIAMOND step itself is not run here: it needs ~3 min on 2 cores and
DIAMOND has no reliable Windows build. Command used, on Linux:

  diamond makedb --in reduced30_train.fasta -d index_db
  diamond blastp -q swissprot_July31_clean.fasta -d index_db -o diamond_hits.tsv \
      --ultra-sensitive --max-target-seqs 25 \
      --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore

A query with no reported alignment is assigned identity 0: DIAMOND failing to
align at --ultra-sensitive means identity below its detection floor (~20-25%),
which belongs in the lowest bin.

Output: stratified_results.json, figures/fig3_accuracy_vs_identity.png

Usage: python scripts/stratify.py
"""

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (depth, truncate, top1, macro_f1, score, bootstrap_ci,
                      build_pairs)

from paths import DATASETS, RESULTS, FIGURES, need

HITS = os.path.join(RESULTS, "diamond_hits.tsv")
PRED = os.path.join(RESULTS, "july31_annos.json")
TRUE = os.path.join(DATASETS, "swissprot_July31_cutoff_id_to_ec.json")
OUT = os.path.join(RESULTS, "stratified_results.json")
FIGDIR = FIGURES

SEED = 0
BINS = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100.01)]
BIN_NAMES = ["<30%", "30-50%", "50-70%", "70-90%", ">=90%"]
LEVELS = [1, 2, 3, 4]

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def best_identity():
    """Identite maximale de chaque requete a une sequence de l'index."""
    best = defaultdict(float)
    best_hit = {}
    with open(HITS) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            q, s, pident = parts[0], parts[1], float(parts[2])
            if pident > best[q]:
                best[q] = pident
                best_hit[q] = s
    return best, best_hit


def bin_of(identity):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= identity < hi:
            return i
    return len(BINS) - 1


def main():
    pred = json.load(open(need(PRED)))
    true = json.load(open(need(TRUE)))
    need(HITS, "Sortie DIAMOND ; voir l'en-tete de ce fichier.")
    best, best_hit = best_identity()
    rng = random.Random(SEED)

    ident = {q: best.get(q, 0.0) for q in true}
    groups = defaultdict(list)
    for q, v in ident.items():
        groups[bin_of(v)].append(q)

    near = sorted(((v, q, best_hit.get(q)) for q, v in ident.items() if v >= 90),
                  reverse=True)

    results = {
        "n_queries": len(true),
        "n_with_alignment": len(best),
        "n_identity_ge_99": sum(1 for v in ident.values() if v >= 99),
        "n_identity_eq_100": sum(1 for v in ident.values() if v >= 99.9),
        "near_duplicates": [{"query": q, "pident": v, "index_hit": h}
                            for v, q, h in near],
        "bins": {},
    }

    for i, name in enumerate(BIN_NAMES):
        ids = set(groups.get(i, []))
        sub_true = {k: v for k, v in true.items() if k in ids}
        entry = {"n_proteins": len(ids), "levels": {}}
        for level in LEVELS:
            pairs = build_pairs(pred, sub_true, level, "B")
            s = score(pairs)
            if s["n"] >= 2:
                acc_ci, f1_ci = bootstrap_ci(pairs, rng, n_boot=1000)
                s["accuracy_ci95"], s["macro_f1_ci95"] = acc_ci, f1_ci
            entry["levels"][str(level)] = s
        results["bins"][name] = entry

    json.dump(results, open(OUT, "w"), indent=2)

    hdr = f"{'bin':>8} {'prot':>5} {'niv':>4} {'n':>4} {'exactitude':>11} {'macro-F1':>9}"
    print(f"\n{len(true)} requetes | {len(best)} avec alignement DIAMOND | "
          f"{results['n_identity_eq_100']} identiques a 100% a une entree de l'index\n")
    print(hdr)
    print("-" * len(hdr))
    for name in BIN_NAMES:
        e = results["bins"][name]
        for level in LEVELS:
            s = e["levels"][str(level)]
            acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "   -  "
            f1 = f"{s['macro_f1']:.3f}" if s["macro_f1"] is not None else "   -  "
            print(f"{name:>8} {e['n_proteins']:>5} {level:>4} {s['n']:>4} {acc:>11} {f1:>9}")
        print("-" * len(hdr))

    figure(results)


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

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
        "savefig.facecolor": SURFACE,
    })

    series = [(1, "Level 1 — class", BLUE, "o"),
              (2, "Level 2 — subclass", ORANGE, "s"),
              (3, "Level 3 — sub-subclass", AQUA, "^"),
              (4, "Level 4 — full EC", YELLOW, "D")]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = list(range(len(BIN_NAMES)))

    for level, name, color, marker in series:
        y, xs, lo, hi = [], [], [], []
        for i, b in enumerate(BIN_NAMES):
            s = res["bins"][b]["levels"][str(level)]
            if s["accuracy"] is not None and s["n"] >= 5:
                xs.append(i)
                y.append(s["accuracy"])
                ci = s.get("accuracy_ci95") or (s["accuracy"], s["accuracy"])
                lo.append(s["accuracy"] - ci[0])
                hi.append(ci[1] - s["accuracy"])
        # IC95 montres seulement sur les deux series etiquetees, sinon illisible
        if level in (1, 4):
            ax.errorbar(xs, y, yerr=[lo, hi], color=color, linewidth=2.2,
                        marker=marker, markersize=8, capsize=3.5, elinewidth=1.2,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=name, zorder=3)
        else:
            ax.plot(xs, y, color=color, linewidth=2.2, marker=marker, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name,
                    zorder=2, alpha=0.9)
        # etiquettes selectives : seulement les deux series qui portent le
        # contraste, a chaque extremite de l'axe
        if level in (1, 4):
            dy = 20 if level == 1 else -22
            for j in (0, -1):
                ax.annotate(f"{y[j]:.2f}", xy=(xs[j], y[j]), xytext=(0, dy),
                            textcoords="offset points", color=color, fontsize=9,
                            fontweight="bold", va="center", ha="center")

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_NAMES)
    ax.set_xlim(-0.4, len(BIN_NAMES) - 0.6)
    ax.set_ylabel("Top-1 accuracy", fontsize=10)
    ax.set_xlabel("Maximum sequence identity to the indexed corpus (DIAMOND)",
                  fontsize=9.5, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for i, b in enumerate(BIN_NAMES):
        ax.annotate(f"n={res['bins'][b]['n_proteins']}", xy=(i, 0.03),
                    ha="center", fontsize=8, color=MUTED)

    # errorbar() et plot() n'enregistrent pas dans l'ordre d'appel : on remet
    # la legende dans l'ordre des niveaux
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for _, n, _, _ in series]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="center left", fontsize=9, labelcolor=INK2,
              handlelength=1.6, bbox_to_anchor=(0.02, 0.40))
    ax.set_title("Accuracy holds at the class level but collapses at full EC\n"
                 "once identity to the indexed corpus drops below 50%",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=12)
    fig.text(0.01, 0.015,
             "Swiss-Prot July-31 cutoff, 244 sequences, masked convention. Queries with no DIAMOND "
             "alignment are placed in the lowest bin.\nError bars (levels 1 and 4 only): bootstrap "
             "95% CI, 1,000 resamples. Levels 2 and 3 omit them for legibility.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig3_accuracy_vs_identity.png")
    fig.savefig(out, dpi=200)
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
