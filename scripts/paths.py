"""
Chemins du projet, centralises pour que les scripts tournent depuis scripts/
quel que soit le repertoire courant.

Les gros fichiers (sequences, poids, index FAISS) ne sont pas versionnes.
Par defaut on les cherche dans <repo>/HiFiNN/ ; la variable d'environnement
HIFINN_DATA permet de les placer ailleurs, par exemple sur un disque externe :

    Windows PowerShell :  $env:HIFINN_DATA = "E:\\hifinn_data\\HiFiNN"
    Linux / macOS      :  export HIFINN_DATA=/mnt/data/HiFiNN
"""

import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)

DATA_ROOT = os.environ.get("HIFINN_DATA", os.path.join(REPO, "HiFiNN"))
DATASETS = os.path.join(DATA_ROOT, "Datasets")
MODELDATA = os.path.join(DATA_ROOT, "ModelData")

RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")

# Embeddings ESM-2 : ~500 Mo pour 244 sequences, souvent hors du depot.
EMBEDDINGS = os.environ.get("HIFINN_EMBEDDINGS",
                            os.path.join(DATA_ROOT, "esm_embeddings_july31"))

for _d in (RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)


def need(path, hint=""):
    """Message clair quand un fichier non versionne manque."""
    if not os.path.exists(path):
        raise SystemExit(
            f"Fichier introuvable : {path}\n"
            f"{hint}\n"
            f"Racine de donnees actuelle : {DATA_ROOT}\n"
            f"(definir HIFINN_DATA pour la changer)"
        )
    return path
