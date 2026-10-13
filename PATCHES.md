# Changes made to the released code

Four changes to the HiFi-NN release, kept as small as possible. Three fix
defects; one adds a configuration key. None alters the model, the index or the
scoring.

The upstream files live under `HiFiNN/hifinn-hifinn_final/` and are not
redistributed here — apply these to your own copy of the release.

---

## 1. Sanitise FASTA headers before embedding

**File:** none — a preprocessing step.

`esm/scripts/extract.py` names each output tensor after the whole header line.
The shipped `swissprot_July31_cutoff_ec.fasta` has headers of the form:

```
>A0A8I5ZNK2 <unknown description>
```

`<`, `>` and spaces are illegal in Windows filenames, so the run fails on the
first batch with `RuntimeError: open file failed with error code: 123`
(`ERROR_INVALID_NAME`). Legal on Linux and macOS, hence invisible there.

Rewriting the FASTA with bare accessions is required in any case:
`swissprot_July31_cutoff_id_to_ec.json` is keyed on bare accessions, while
`load_embeddings` derives ids from filename stems, so the two only line up once
headers are stripped.

```python
cur = line[1:].split()[0]   # 'A0A8I5ZNK2 <unknown description>' -> 'A0A8I5ZNK2'
```

All 244 accessions then match the ground-truth keys exactly: no orphan sequence,
no orphan label.

**Related, not fixed:** `load_embeddings` in `utils/file_utils.py` uses
`fp.strip(".pt")`, which strips *characters*, not a suffix. An id ending in `p`,
`t` or `.` would be over-trimmed. No accession in this test set is affected, but
`fp[:-3]` or `os.path.splitext` would be safer.

---

## 2. `torch.load` under PyTorch 2.6+

**File:** `utils/file_utils.py`, lines 163, 169, 173, 197, 215.

```python
# before
ckpt = torch.load(model_path, map_location=torch.device("cpu"))

# after
ckpt = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
```

Since PyTorch 2.6 the default is `weights_only=True`; the released checkpoints
contain `omegaconf.listconfig.ListConfig` and `numpy._core.multiarray._reconstruct`,
which that mode refuses. Safe for a trusted release, not for arbitrary
checkpoints.

---

## 3. Make the HiFi-NN batch size configurable

**Files:** `annotate.py`, `configs/annotate.yaml`.

`embed_queries` defaults to `batch_size=32` and `annotate.py` never overrides it.
`batch_collate_fn` pads each batch to its longest sequence, so 32 sequences at
1,022 residues allocate a `32 x 20 x 1022^2` float32 attention tensor in
`MultiheadAttention.forward` — one 2.67 GB allocation, which fails on 16 GB:

```
RuntimeError: [enforce fail at alloc_cpu.cpp:117] DefaultCPUAllocator:
not enough memory: you tried to allocate 2673879040 bytes
```

```python
# annotate.py, 'folder' branch
     queries, query_ids = embed_queries(
         config["input"],
         model,
         representations="representations",
         layer=32,
         padding_value=0,
         device=device,
+        batch_size=config.get("batch_size", 4),
     )
```

```yaml
# configs/annotate.yaml
batch_size: 4
```

At 4 the peak is about 330 MB. Results are unchanged: batching affects only
padding, and padded positions are masked.

---

## 4. Numerically stable confidence softmax

**File:** `annotators/nearest_neighbours.py`, in `get_confidence_scores` and
again in `get_confidence_and_distance`.

```python
# before
denominator = np.sum([np.exp((-d) / T) for d in distances[i]])
...
label_to_prob[label] += np.exp(-distances[i][j] / T) / denominator

# after
d_min = float(np.min(distances[i]))
denominator = np.sum([np.exp(-(d - d_min) / T) for d in distances[i]])
...
label_to_prob[label] += np.exp(-(distances[i][j] - d_min) / T) / denominator
```

With `T = 0.001`, `exp(-d/T) = exp(-1000d)`. Once every one of the k distances
exceeds roughly 0.75, all terms underflow to zero in float64, the denominator is
zero, and every confidence for that query is `NaN`.

On `swissprot_July31_cutoff` this affected **56 of 244 queries (23%)** — by
construction exactly the queries with no close neighbour, i.e. the remote
homologues a benchmark most needs to measure. The shift is the standard
log-sum-exp trick: algebraically identical, and it removes every NaN.

Ranking is unaffected, since the shift is constant within a query. What changes
is that a confidence exists at all for remote queries.

---

## Not a code change, but required

`get_model_embeddings_from_dl` writes intermediate embeddings to
`./hifinn_embeddings/` and then reloads the **entire directory**. A previous
run's files are silently included in the next run's results. Clear it between
runs:

```powershell
Remove-Item .\hifinn_embeddings\* -Force
```

This produces no error — only wrong numbers — and is the easiest of these to
miss.
