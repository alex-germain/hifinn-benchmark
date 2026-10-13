# Running notes

Six environment obstacles hit while getting the HiFi-NN release to run, none of
them defects in the code itself. They are recorded because together they are the
difference between "the release works" and "the release works on the machine it
was written on", and because each cost an hour that someone else need not spend.

Code defects are in [PATCHES.md](PATCHES.md); results are in
[README.md](README.md).

**Environment.** Windows 11, 16 GB RAM, no GPU, no admin rights. Miniconda,
Python 3.10, PyTorch 2.14 CPU, faiss-cpu, fair-esm. Model weights and FAISS
index from the Zenodo release.

---

## 1. The shipped conda environment cannot be created

`conda env create -f requirements.yaml` fails with:

```
PackagesNotFoundInChannelsError: biopython==1.78=py312h80987f9_0
```

The file pins **exact build hashes**, and `py312h80987f9_0` is a macOS
arm64 build. The environment is therefore only creatable on the machine family
it was exported from. `requirements.txt` has the same problem in a milder form.

*Worked instead:*

```bash
conda create -n hifinn python=3.10
conda activate hifinn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning biopython numpy pyyaml tqdm einops \
            faiss-cpu fair-esm omegaconf hydra-core matplotlib
```

Exporting with `conda env export --no-builds`, or `--from-history`, would make
the shipped file portable.

## 2. conda is not on PATH after a user-scope install

Installing Miniconda without admin rights leaves `conda` unavailable in a fresh
shell. The executable is at `%USERPROFILE%\miniconda3\Scripts\conda.exe`; running
`conda init powershell` once and restarting the shell fixes it.

A separate first-run block:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

## 3. PyTorch 2.6 changed the default of `torch.load`

Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`. The released
checkpoints contain `omegaconf.listconfig.ListConfig` (hyperparameters) and
`numpy._core.multiarray._reconstruct`, so loading raises:

```
UnpicklingError: Weights only load failed
```

Five calls in `utils/file_utils.py` need `weights_only=False` — lines 163, 169,
173, 197 and 215. This is safe here because the checkpoints come from a trusted
release; it would not be safe for an arbitrary downloaded checkpoint.

## 4. Duplicate OpenMP runtimes, and why `KMP_DUPLICATE_LIB_OK` is not a fix

Symptom: the process dies with exit code `-1073741819` (`0xC0000005`, access
violation), with no traceback and no output. Intermittently — the same command
succeeded, failed at model load, and failed mid-batch on three consecutive runs.

Cause: conda's MKL and PyTorch each ship an OpenMP runtime, and both are loaded
into the process. `KMP_DUPLICATE_LIB_OK=TRUE` is widely recommended, and is what
made the pipeline run here, but Intel documents it as *unsafe — may cause
crashes or silently incorrect results*. It suppresses the guard rather than
removing the conflict, which is why crashes continued at random points.

What actually stabilised it: `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`,
removing the thread contention that triggers the fault. Slower, and reliable.

The clean fix, not applied here, is an environment with a single OpenMP runtime —
pip-only, with no conda MKL packages.

Two further mitigations worth keeping:

- `scripts/embed_july31.py` processes one sequence at a time and **skips files
  already written**, so a crash costs seconds rather than the whole run.
- `python -u` disables output buffering. Without it, a hard crash loses whatever
  was buffered, so the run appears to die silently before its first print. Two
  of the failures above looked like crashes at startup and were in fact crashes
  several batches in.

## 5. git is not installed, and installing it needs admin

`annotate.py` shells out to `./esm/scripts/extract.py`, so the ESM repository has
to be present — `pip install fair-esm` provides the importable package but not
the script. Cloning needs git; installing git needed elevation that was not
available.

*Worked instead:*

```powershell
Invoke-WebRequest -Uri https://github.com/facebookresearch/esm/archive/refs/heads/main.zip -OutFile esm.zip
tar -xf esm.zip
Rename-Item esm-main esm
```

## 6. The ESM subprocess crashes only when spawned from `annotate.py`

`esm_embed` in `utils/embedding_utils.py` runs the embedding script through
`subprocess.run`. That child process died with `3221225477` every time, while the
identical command run directly from the shell completed normally. The subprocess
inherits an environment in which the OpenMP conflict of section 4 is already
live.

*Worked instead:* run the embedding step separately and point `annotate.py` at
the resulting directory. `check_input_format` already supports a folder of
pre-computed embeddings, which makes this a supported path rather than a hack:

```yaml
input: /path/to/esm_embeddings/
```

`scripts/embed_july31.py` is that separate step.

## 7. Disk space

Not an obstacle of the release's making, but worth stating as a requirement: the
model weights (258 MB), the FAISS index (562 MB), `reduced30_train.fasta`
(118 MB), the ESM-2 650M checkpoint (2.5 GB, downloaded on first use) and ~500 MB
of per-residue embeddings for 244 sequences come to roughly **4 GB** before any
output.

`TORCH_HOME` relocates the ESM checkpoint cache; `HIFINN_DATA` and
`HIFINN_EMBEDDINGS` (see `scripts/paths.py`) relocate the rest. All three were
needed here.

---

## Summary

| # | Obstacle | Fix |
|---|---|---|
| 1 | `requirements.yaml` pins platform-specific build hashes | build the environment manually |
| 2 | conda absent from PATH; ToS prompt blocks non-interactive use | `conda init`, `conda tos accept` |
| 3 | `torch.load` defaults to `weights_only=True` since PyTorch 2.6 | `weights_only=False` on 5 calls |
| 4 | duplicate OpenMP runtimes cause intermittent `0xC0000005` | `OMP_NUM_THREADS=1`; resumable script; `python -u` |
| 5 | `annotate.py` needs the ESM repo, and git needs admin | download the ZIP |
| 6 | the ESM subprocess crashes only when spawned | run it separately, use folder input |
| 7 | ~4 GB of artefacts | `TORCH_HOME`, `HIFINN_DATA`, `HIFINN_EMBEDDINGS` |
