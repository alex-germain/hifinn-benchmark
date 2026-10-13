# Embeddings ESM-2 pour swissprot_July31, robuste aux crashs OpenMP sous Windows.
# - un seul thread (evite le crash 0xC0000005 du double runtime OpenMP)
# - une sequence a la fois (empreinte memoire minimale)
# - reprend ou il s'est arrete : relancer le script apres un crash suffit
#
# Sortie : un .pt par sequence dans HIFINN_EMBEDDINGS (~500 Mo pour 244).
# A lancer depuis la racine du depot :
#   python -u scripts/embed_july31.py

import os
import sys

import torch
import esm
from esm.data import FastaBatchedDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATASETS, EMBEDDINGS, need

FASTA = need(os.path.join(DATASETS, "swissprot_July31_clean.fasta"),
             "FASTA aux en-tetes nettoyes ; voir PATCHES.md section 1.")
OUT = EMBEDDINGS
LAYER = 32
TRUNC = 1022

torch.set_num_threads(1)
os.makedirs(OUT, exist_ok=True)

print("chargement du modele...", flush=True)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
batch_converter = alphabet.get_batch_converter(TRUNC)

ds = FastaBatchedDataset.from_file(FASTA)
total = len(ds)
print(f"{total} sequences lues dans {FASTA}", flush=True)

ecrits, sautes = 0, 0
for label, seq in zip(ds.sequence_labels, ds.sequence_strs):
    lab = label.split()[0]
    path = os.path.join(OUT, lab + ".pt")
    if os.path.exists(path):
        sautes += 1
        continue

    _, _, toks = batch_converter([(lab, seq)])
    with torch.no_grad():
        out = model(toks, repr_layers=[LAYER], return_contacts=False)
    n = min(len(seq), TRUNC)
    rep = out["representations"][LAYER][0, 1 : n + 1].clone()
    torch.save({"label": lab, "representations": {LAYER: rep}}, path)

    ecrits += 1
    print(f"[{ecrits + sautes}/{total}] {lab}  L={rep.shape[0]}", flush=True)

print(f"TERMINE — {ecrits} ecrits, {sautes} deja presents, {ecrits + sautes}/{total}", flush=True)
