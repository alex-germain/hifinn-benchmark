"""
Evaluation de HiFi-NN sur swissprot_July31_cutoff.

Produit, pour chaque niveau de la hierarchie EC (1 a 4) :
  - trois conventions de traitement des annotations partielles (voir ci-dessous)
  - l'exactitude top-1 et le macro-F1
  - deux lignes de base nulles (classe majoritaire de l'index, tirage pondere)

Conventions pour une verite partielle du type '4.-.-.-' :
  A  stricte   : la verite est tronquee litteralement ('4.-' au niveau 2),
                 donc une prediction '4.2' compte comme une erreur.
                 Mesure l'accord litteral avec la base telle qu'elle est ecrite.
  B  masquee   : la proteine n'est evaluee qu'aux niveaux ou sa verite est
                 specifiee. Convention standard (type CAFA). Chiffre principal.
  C  indulgente: une prediction compatible avec la verite partielle (meme
                 prefixe) compte comme correcte.

Usage : python scripts/evaluate.py
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import DATASETS, MODELDATA, RESULTS, need

PRED_PATH = os.path.join(RESULTS, "july31_annos.json")
TRUE_PATH = os.path.join(DATASETS, "swissprot_July31_cutoff_id_to_ec.json")
INDEX_EC_PATH = os.path.join(MODELDATA, "swissprot_all_ec_nums.json")
OUT_PATH = os.path.join(RESULTS, "evaluation_results.json")

SEED = 0


def depth(ec):
    """Nombre de niveaux reellement specifies avant le premier '-'."""
    d = 0
    for part in ec.split("."):
        if part == "-":
            break
        d += 1
    return d


def truncate(ec, level):
    """Tronque un EC au niveau demande, en gardant les '-' tels quels."""
    return ".".join(ec.split(".")[:level])


def top1(pred_dict):
    """EC de plus forte confiance."""
    return max(pred_dict.items(), key=lambda kv: kv[1])[0]


def macro_f1(pairs):
    """
    pairs : liste de (ensemble_de_verites, prediction) au meme niveau.
    Macro-F1 moyenne sur les classes presentes dans la verite.
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    classes = set()
    for truths, pred in pairs:
        classes |= truths
        if pred in truths:
            tp[pred] += 1
            for t in truths - {pred}:
                fn[t] += 1
        else:
            fp[pred] += 1
            for t in truths:
                fn[t] += 1
    f1s = []
    for c in sorted(classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return (sum(f1s) / len(f1s)) if f1s else 0.0, len(classes)


def build_pairs(pred, true, level, convention):
    """Construit la liste (verites, prediction) au niveau donne, selon la convention."""
    pairs = []
    for pid, truths in true.items():
        if pid not in pred or not pred[pid]:
            continue
        p = truncate(top1(pred[pid]), level)
        d = max(depth(e) for e in truths)

        if convention == "A":
            t = {truncate(e, level) for e in truths}
        elif convention == "B":
            if d < level:
                continue  # verite non specifiee a ce niveau : exclue
            t = {truncate(e, level) for e in truths if depth(e) >= level}
        elif convention == "C":
            if d < level:
                # compatible = meme prefixe jusqu'a la profondeur connue
                t = {truncate(e, level) for e in truths}
                if any(truncate(top1(pred[pid]), depth(e)) == truncate(e, depth(e))
                       for e in truths):
                    t = {p}  # compte comme correct
            else:
                t = {truncate(e, level) for e in truths if depth(e) >= level}
        else:
            raise ValueError(convention)

        if t:
            pairs.append((t, p))
    return pairs


def score(pairs):
    if not pairs:
        return {"n": 0, "accuracy": None, "macro_f1": None, "n_classes": 0}
    acc = sum(1 for t, p in pairs if p in t) / len(pairs)
    f1, nc = macro_f1(pairs)
    return {"n": len(pairs), "accuracy": acc, "macro_f1": f1, "n_classes": nc}


def bootstrap_ci(pairs, rng, n_boot=2000, alpha=0.05):
    """
    Intervalle de confiance par bootstrap sur les proteines (reechantillonnage
    avec remise). Retourne (bas, haut) pour l'exactitude et le macro-F1.
    Necessaire au niveau 4, ou ~100 classes se partagent ~150 proteines.
    """
    if len(pairs) < 2:
        return None, None
    accs, f1s = [], []
    n = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(1 for t, p in sample if p in t) / n)
        f1s.append(macro_f1(sample)[0])
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    accs.sort()
    f1s.sort()
    return (accs[lo_i], accs[hi_i]), (f1s[lo_i], f1s[hi_i])


def baselines(true, index_ec, level, rng):
    """
    Lignes de base nulles, evaluees sous la convention B.
    - majority : toujours la classe la plus frequente de l'index
    - weighted : tirage aleatoire pondere par la frequence dans l'index
    """
    counts = Counter()
    for ecs in index_ec.values():
        for e in ecs:
            if depth(e) >= level:
                counts[truncate(e, level)] += 1
    if not counts:
        return {}
    majority = counts.most_common(1)[0][0]
    labels = list(counts)
    weights = [counts[c] for c in labels]

    maj_pairs, targets = [], []
    for pid, truths in true.items():
        d = max(depth(e) for e in truths)
        if d < level:
            continue
        t = {truncate(e, level) for e in truths if depth(e) >= level}
        maj_pairs.append((t, majority))
        targets.append(t)

    # Le tirage pondere est moyenne sur plusieurs realisations : un seul tirage
    # par requete donne un chiffre reproductible mais bruite.
    n_draws = 200
    accs, f1s = [], []
    for _ in range(n_draws):
        draw = [(t, rng.choices(labels, weights=weights, k=1)[0]) for t in targets]
        accs.append(sum(1 for tt, p in draw if p in tt) / len(draw))
        f1s.append(macro_f1(draw)[0])
    rnd = {"n": len(targets), "n_classes": len(labels),
           "accuracy": sum(accs) / n_draws, "macro_f1": sum(f1s) / n_draws,
           "n_draws": n_draws}
    return {"majority": score(maj_pairs), "weighted_random": rnd}


def main():
    pred = json.load(open(need(PRED_PATH, "Lancer annotate.py, puis copier sa sortie dans results/.")))
    true = json.load(open(need(TRUE_PATH, "Jeu de test fourni avec la release.")))
    index_ec = json.load(open(need(INDEX_EC_PATH, "Fourni avec la release (ModelData).")))
    rng = random.Random(SEED)

    results = {"n_queries": len(true), "levels": {}}
    for level in range(1, 5):
        entry = {"conventions": {}}
        for conv in ("A", "B", "C"):
            pairs = build_pairs(pred, true, level, conv)
            entry["conventions"][conv] = score(pairs)
            if conv == "B":
                acc_ci, f1_ci = bootstrap_ci(pairs, rng)
                entry["conventions"][conv]["accuracy_ci95"] = acc_ci
                entry["conventions"][conv]["macro_f1_ci95"] = f1_ci
        entry["baselines"] = baselines(true, index_ec, level, rng)
        results["levels"][str(level)] = entry

    json.dump(results, open(OUT_PATH, "w"), indent=2)

    # affichage
    print(f"\n{len(true)} requetes, index de {len(index_ec)} entrees\n")
    hdr = f"{'niv':>4} {'conv':>5} {'n':>5} {'exactitude':>11} {'macro-F1':>9} {'classes':>8}"
    print(hdr)
    print("-" * len(hdr))
    for level in range(1, 5):
        e = results["levels"][str(level)]
        for conv in ("A", "B", "C"):
            s = e["conventions"][conv]
            line = (f"{level:>4} {conv:>5} {s['n']:>5} {s['accuracy']:>11.3f} "
                    f"{s['macro_f1']:>9.3f} {s['n_classes']:>8}")
            if s.get("accuracy_ci95"):
                a, f = s["accuracy_ci95"], s["macro_f1_ci95"]
                line += f"   IC95 exact. [{a[0]:.3f}, {a[1]:.3f}]  F1 [{f[0]:.3f}, {f[1]:.3f}]"
            print(line)
        for name, s in e["baselines"].items():
            print(f"{level:>4} {name[:5]:>5} {s['n']:>5} {s['accuracy']:>11.3f} "
                  f"{s['macro_f1']:>9.3f} {s['n_classes']:>8}")
        print("-" * len(hdr))
    print(f"\nEcrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
