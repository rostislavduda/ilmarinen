# Ilmarinen — a physics-principled neural-architecture meta-optimizer

[![CI](https://github.com/rostislavduda/ilmarinen/actions/workflows/ci.yml/badge.svg)](https://github.com/rostislavduda/ilmarinen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rostislavduda/ilmarinen/branch/main/graph/badge.svg)](https://codecov.io/gh/rostislavduda/ilmarinen)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21542050.svg)](https://doi.org/10.5281/zenodo.21542050)

`ilmarinen` selects a neural architecture — its computational *contract* (arena), its primitive
operations, and its width and depth — for a dataset, by minimizing a single description-length
objective. Every choice is priced as *risk + complexity*, `J = R + mu * Omega`, with each complexity
charge `Omega` a **derived** code length rather than a hand-set constant. The organizing idea is a
physicist's ordering: fix the kinematics (symmetry, effective dimension) first, then let the dynamics
(training) select the remaining degrees of freedom under one Occam objective.

This is a research package. Findings — including first-class negatives — are written up in three PDF
reports (see **Reports** below). Detailed per-study write-ups are kept in the author's working tree
and are not part of this distribution.

## Install / instantiate a container

Two environment files are provided so a container can be brought up in one command.

**conda** (recommended for the heavy binary deps — torch, rdkit):

```bash
conda env create -f environment.yml
conda activate ilmarinen
python -m ilmarinen._selfcheck            # -> "INTEGRITY OK: all modules and primitives present"
```

**pip:**

```bash
pip install -r requirements.txt         # core runtime + dataset/model extras
python -m ilmarinen._selfcheck
```

The package runs on **CPU** out of the box and uses a **GPU** (CUDA or Apple-Silicon MPS) when asked —
see **Device** below. Python 3.11/3.12 recommended. For a minimal image, only the CORE block of `requirements.txt`
(`numpy`, `scipy`, `torch`) is needed to import `ilmarinen` and run a fit on in-memory tensors; the
dataset/model extras are imported lazily and only when a given dataset or primitive is used.

**Running from a tarball** (no install): unpack and run in place — `python -m ilmarinen._selfcheck`,
then the runners below.

**Docker.** Reproducible **CPU** and **CUDA** images build from this repo:

```bash
docker build -t ilmarinen .                            # CPU (deps pinned via requirements.lock)
docker run --rm ilmarinen                              # -> INTEGRITY OK

docker build -f Dockerfile.cuda -t ilmarinen:cuda .    # CUDA (host needs NVIDIA driver + nvidia-container-toolkit)
docker run --gpus all --rm ilmarinen:cuda python -c "import torch; print(torch.cuda.is_available())"
```

Tagged releases publish both to GHCR: `ghcr.io/rostislavduda/ilmarinen:<version>` / `:latest` (CPU) and
`:<version>-cuda` / `:cuda` (CUDA).

## Quick start

```python
import numpy as np
from ilmarinen import AllGraph, AllData

# in-memory data: a batch of sequences (N, T, C) with integer labels
X = np.random.randn(256, 40, 3).astype("float32")
y = (X[:, :, 0].mean(axis=1) > 0).astype("int64")
data = AllData.dense_tensor(X, y)

mg = AllGraph(width=32, depth=1, epochs=20)
result = mg.fit(data, task="classification", select="gibbs")
print(result["architecture"], result["value"])
```

`AllData` constructors cover every arena: `dense_tensor(X, y)` (grids/sequences),
`point_sets(node_feats, y, positions=...)`, `graphs(node_feats, edges, y, positions=...)`, and
`functions(a, y, grid)` (operators). `AllGraph.route`/`.fit` dispatch to the right contract
automatically; `explain`/`AllGraph.explain` reports the selected architecture as its own explanation.

## Streaming large datasets (train on data bigger than RAM/VRAM)

The `dense_tensor`/`graphs`/`functions` constructors hold the whole dataset in memory. To train on data
that does not fit, wrap it in a **lazy source** and use the streaming constructors — the fit then pulls one
minibatch at a time instead of materializing the dataset:

```python
from ilmarinen import AllGraph, AllData, MemmapDenseSource

source = MemmapDenseSource("images.npy")  # memory-mapped; never fully loaded
data = AllData.dense_stream(source, y=labels, kind_hint="spatial")
AllGraph(width=32, depth=2, epochs=30).fit(data, task="classification", n_out=10)
```

Streaming is **opt-in by constructor** (building the input any other way keeps the exact in-memory path)
and, for the map-style sources, trains to **bit-for-bit identical weights** as the equivalent in-memory fit:

- `AllData.dense_stream(DenseSource, kind_hint=...)` — dense contracts (sequence/spatial/volumetric/4d).
- `AllData.graph_stream(GraphSource, kind_hint="graph"|"equivariant"|"set")` — relational contracts.
- `AllData.functions_stream(OperatorSource)` — the neural-operator contract.

Back a source with anything random-access: `Memmap{Dense,Operator}Source` over `.npy`/`np.memmap`,
`LazyGraphSource(loader)` over per-graph files / HDF5 / a DB, or your own `DenseSource`/`GraphSource`/
`OperatorSource` subclass. Selection (`select_size`, `select="gibbs"`, `tiebreak`, `angular_from_data`) runs
under streaming on a bounded resident subsample (`stream_subsample_cap`, default 20 000) while the winner
deploy-trains on the full stream. Two opt-in add-ons: `MemmapDenseSource(cache_size=…)` (a bounded LRU that
avoids re-reads) and `AllGraph(stream_prefetch=True)` (overlaps the next minibatch's fetch with compute) —
both preserve bit-identity. For a **forward-only** source (no random access, e.g. a network stream), use
`AllData.dense_iter(IterableDenseSource, kind_hint=…, n_out=…)`; it uses a seeded windowed shuffle buffer and
a hash-based train/val split — deterministic given the seed, but not identical to the in-memory fit. Runnable
end-to-end examples: `python -m examples.streaming_dataset` (and `…streaming_graphs` / `…streaming_operator`).

## Device (CPU / GPU)

`AllGraph(device=...)` accepts `"cpu"` (default), `"cuda"`, `"mps"`, or `"auto"` (picks CUDA >
Apple-Silicon MPS > CPU). On **Apple Silicon**, the contracts whose fit is launch- or scatter-bound —
`sequence`, `volumetric`, `4d`, and the relational `graph`/`equivariant`/`set` — are automatically pinned
to the CPU, where they run **1.4–4× faster** than MPS at this package's model sizes; `spatial` and
`operator` stay on the MPS GPU (**2–5× faster** there). This routing is MPS-only — on **CUDA** every
contract runs on the GPU. The same policy applies to a model reloaded with `AllGraph.load(...)`. Set
`PYTORCH_ENABLE_MPS_FALLBACK=1` for the most robust MPS path (any op lacking an MPS kernel then runs on CPU).

## What it does

**Eight computational contracts (arenas).** `sequence`, `spatial`, `volumetric`, `4d`, `graph`,
`equivariant`, `set`, `operator` — one meta-router over all eight. The contract choice near a fit tie
is itself folded into `J = R + mu * Omega` with a derived structural code length `Omega_struct`.

**One priced-selection ladder.** Contract, primitive, depth, width, and weights are all rungs of the
same MDL functional — discrete at the top (contract, and the primitive argmax), continuous below.
Width and depth are chosen by a marginal-value rule; the primitive by a Gibbs readout over
clean-solo energies; the mixture (optionally) by a sparsity-priced `alpha`.

**A symmetry-discovery front-end.** Continuous symmetry via the Lie-derivative nullspace, discrete
symmetry via equivariance testing, enforcement via a commutant equivariant layer, with false-positive
guards — the kinematic rung that removes exact redundancy before training.

**Faithful-by-construction read-outs** (interpretability as a corollary, all opt-in, none change the
selection):

| flag | what it reports |
|---|---|
| `report_llc` | the local learning coefficient (RLCT) `lambda` at the converged optimum |
| `price_singular` | fuse the functional code length `lambda*log n` into the contract charge (guarded) |
| `developmental_llc` | the complexity trajectory `lambda(t)` over training (staged-learning onsets) |
| `report_thermo` | the single free-energy form's three-level temperature hierarchy, with a consistency guard |
| `report_response` | the curvature of the selection (readout specific heat + first-order contract transitions) |
| `report_ledger` | the effective-dimension ledger: one participation-ratio functional across the coarse-graining axis, plus `lambda` |
| `contract_posterior`, `price_equivariance`, `price_modes` | a contract posterior; priced approximate equivariance; priced spectral-mode selection |

## Running the validation suites

```bash
python -m validation_runners.run_quick_validation      # quick multi-dataset run
python -m validation_runners.run_standard_validation   # the fuller suite

# any read-out/pricing flag can be turned on, e.g.:
python -m validation_runners.run_quick_validation --only QM7-equiv --tiebreak --price_singular
python -m validation_runners.run_quick_validation --report_llc --report_ledger --report_response
```

Both runners expose every opt-in flag (`--help` lists them). `studies/` holds the standalone study
reproducers.

The standard runner accumulates its results into one merge-on-write JSON
(`$ILMARINEN_DATA_DIR/standard_val_rows.json`), so the suite can be run in batches — each `--contracts`
invocation upserts its rows into the same document, and rows are flushed after every dataset.
`validation_runners/make_results_table.py` renders that document as the table below.

### Bring your own dataset (Kaggle)

The runners above train on the curated registry. `run_kaggle_validation.py` instead takes an **arbitrary
Kaggle handle**, ingests it, and hands the result to the *same* pipeline — it imports the shared flags,
model construction, and evaluation from `run_standard_validation`, so it is configured identically:

```bash
pip install "ilmarinen[kaggle]"          # kagglehub + pandas + pillow (lazy, per mode)

# ALWAYS inspect first: files, per-column verdicts, target/task guesses, split, chance, route — no training
python validation_runners/run_kaggle_validation.py --handle uciml/iris --inspect
python validation_runners/run_kaggle_validation.py --handle uciml/iris --epochs 40
python validation_runners/run_kaggle_validation.py --local_dir /path/to/extracted   # offline, no Kaggle
```

Three auto-detected modes: a **table** (numeric + one-hot matrix → rank 2 → the `sequence` contract), an
**image class-directory tree** (→ `(N,C,hw,hw)` → `spatial`), or a raw **`.npy`/`.npz`** (rank 2–6 passed
straight to the grid-rank router). The split is drawn first and every statistic — imputation median,
z-score, one-hot levels, label map, per-channel image stats — is fit on the train rows only.

Read `skill`, not accuracy: `chance` is the *train-majority* predictor scored on the test split. And note
that a flat tabular vector is read as a length-`n_features` sequence — ilmarinen is a meta-optimizer over
computational contracts, not a tabular specialist, so **expect gradient boosting to win on tabular Kaggle
data**. The module docstring lists all the caveats. Kaggle rows are written to a separate
`kaggle_val_rows.json` and never enter the benchmark table below.

## Standard validation suite — results

Every dataset in the registry, at **full size** (`reduced=False`: the model trains on the entire train
split and is scored on the entire held-out test split), routed and sized by the meta-optimizer itself —
no per-dataset hand-tuning. The architecture column is what the priced-selection ladder *chose*; the
parameter count is the deployed net after sparse selection.

Protocol:

- **Selection.** `--select sparse` (sparsity-priced mixture, `--sparsity_mu 0.3`) with
  `--select_size variable` (variable-width-per-layer + emergent depth), on top of `--preset opt` — the
  data-size-robust per-contract processing flags from the flag search. Passing `--select sparse`
  explicitly keeps the sparse readout everywhere, including on the operator contract where `opt` would
  otherwise select the Gibbs readout.
- **Training length.** Each deployed model starts on the contract's 100-epoch budget and is granted a
  further 100 epochs whenever it exhausts that budget *without* meeting the convergence criterion,
  continuing from the same weights and optimizer state, up to a 1000-epoch ceiling. Convergence is the
  plateau test: the monitored loss failing to improve by ≥1% relative for 10 consecutive epochs
  (`--auto_epoch val --auto_epoch_patience 10 --auto_epoch_min_epochs 10`), after which the
  best-scoring epoch's weights are restored (`--auto_epoch_restore_best`). A row daggered in the epochs
  column hit the ceiling instead and is therefore budget-limited, not converged.
- **Monitor caveat.** `--auto_epoch val` holds out ~15% of the training data, but only when that leaves
  a reliable monitor (≥50 held-out samples at ≤35% of the data); smaller datasets fall back to
  monitoring training loss automatically. The per-dataset monitor actually used is recorded as
  `auto_epoch_monitor` in the results JSON.
- **Skill** is the cross-dataset comparable axis: `(acc − chance)/(1 − chance)` for classification, `R²`
  for regression, AUC where the dataset's headline metric is AUC.
- SOTA references are the registry's own per-dataset citations, abridged to their headline figure; the
  full strings live in `ilmarinen/core/dataset_registry.py` and `extended_datasets.py`. They come from
  the literature on each dataset, not from re-runs here, and the comparison is *not* like-for-like on
  budget: these are single-seed, single-configuration runs of a general meta-optimizer against
  per-dataset specialist architectures.

<!-- BEGIN:stdval -->

**`sequence`** -- 1D series (UCR/UEA, epidemiological, tabular)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| ItalyPowerDemand | acc | 0.9660 | ~0.97 (ROCKET/InceptionTime) | +0.9320 | `conv→norm` | 7,054 | 1000† |
| ECG5000 | acc | 0.9287 | ~0.94-0.95 (ResNet/InceptionTime/TCN) | +0.9108 | `gated→norm→dilconv` | 25,786 | 47 |
| GunPoint | acc | 0.9467 | ~0.99 (HIVE-COTE/ROCKET) | +0.8933 | `attention→dense→attention` | 60,673 | 45 |
| Superconductivity | R2 | 0.8089 | ~0.92 (XGBoost, Hamidieh 2018) | +0.8089 | `attention→dense→plain` | 114,862 | 20 |
| BasicMotions | acc | 0.8000 | ~1.0 (ROCKET/multivariate) | +0.7333 | `attention→attention→attention` | 19,577 | 546 |
| ACSF1 | acc | 0.5600 | ~0.88 (ROCKET/MultiRocket) | +0.5111 | `conv→dense→conv` | 299,111 | 58 |
| EnglandCovid | R2 | 0.4337 | n/a (no canonical value; MSE ~0.5-0.9 z-scored, lag-dependent in follow-ups) | +0.4337 | `conv→dense→dense` | 94,646 | 1000† |
| OSULeaf | acc | 0.4876 | ~0.97 (HIVE-COTE 2.0), ~0.96 (MultiRocket), ~0.94 (ROCKET) | +0.3851 | `lstm→spectral→attention` | 40,879 | 66 |
| Chickenpox | R2 | 0.3109 | ~0 (recurrent GNNs ~ mean predictor; MSE ~1.1 on z-scored targets, PyG-Temporal) | +0.3109 | `conv→dense→dense→gated` | 400,793 | 22 |

**`spatial`** -- 2D grids (images)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| MNIST | acc | 0.9918 | ~0.99+ (any CNN) | +0.9909 | `pointwise→atrous→atrous` | 297,759 | 21 |
| BloodMNIST | acc | 0.9421 <br><sub>ROC-AUC 0.9967</sub> | ~0.958 (ResNet-18@28), ~0.966 (AutoML) | +0.9339 | `pointwise→pointwise→atrous` | 298,269 | 20 |
| MNISTAngle | R2 | 0.7875 | n/a (semi-synthetic; -> high for a CNN reading digit orientation) | +0.7875 | `pointwise→pointwise→norm` | 297,462 | 15 |

**`volumetric`** -- 3D grids (medical volumes)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| DiffusionBlob3D | R2 | 0.9700 | n/a (synthetic-from-solver; -> 1 for a sufficient 3D conv) | +0.9700 | `norm→conv3d→conv_dw` | 93,144 | 24 |
| OrganMNIST3D | acc | 0.9279 <br><sub>ROC-AUC 0.9958</sub> | ~0.907 (ResNet-18+3D; MedMNIST v2 benchmark best) | +0.9207 | `norm→conv_dw→conv_dw` | 121,498 | 40 |
| VesselMNIST3D | ROC-AUC | 0.6781 <br><sub>acc 0.8770</sub> | ~0.87 (ResNet-18+3D), ~0.93 (best, ACS conv) | +0.6781 | `norm→conv_dw→conv_dw` | 121,345 | 31 |
| SynapseMNIST3D | ROC-AUC | 0.5083 <br><sub>acc 0.6875</sub> | ~0.82 (ResNet-18+3D), ~0.85 (best, ResNet-50+3D) | +0.5083 | `norm→conv_dw→conv_dw` | 121,345 | 23 |

**`4d`** -- 3D+time grids (solver-generated fields)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| AdvectionDiffusion4D | acc | 1.0000 | n/a (synthetic-from-solver; -> 1 for a sufficient 4d model) | +1.0000 | `norm→conv4d→conv4d` | 35,010 | 46 |
| HeatDiffusion3D | R2 | 0.9680 | n/a (synthetic-from-solver; -> 1 for a sufficient 4d model) | +0.9680 | `pointwise→conv4d→conv4d` | 46,902 | 72 |

**`graph`** -- molecular and social graphs

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| Tox21 | ROC-AUC | 0.8037 <br><sub>acc 0.9682</sub> | ~0.75-0.83 (GNN) | +0.8037 | `gat→gcn→gin` | 88,423 | 44 |
| ESOL | MAE[log mol/L] ↓ | 0.6648 <br><sub>R2 0.7854</sub> | ~0.40-0.45 log mol/L (D-MPNN/best GNN, random split) | +0.7854 | `gat→gcn→gcn` | 210,636 | 67 |
| IMDB-BINARY | acc | 0.7350 | ~0.70-0.76 (GIN / graph-kernel SOTA) | +0.4700 | `gat→gcn→gcn→gcn` | 142,942 | 15 |

**`equivariant`** -- E(3)/SO(3) point clouds (quantum chemistry, shapes)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| QM9 ‡ | MAE[meV] ↓ | 101331.5391 <br><sub>R2 0.9835</sub> | ~5-15 meV (SchNet/PaiNN/DimeNet) | +0.9835 | `e_norm→e_painn→e_norm` | 24,985 | 51 |
| QM7 | MAE[kcal/mol] ↓ | 31.0945 <br><sub>R2 0.9655</sub> | < 1 kcal/mol = chemical accuracy (SchNet/PaiNN) | +0.9655 | `e_norm→e_norm→e_gate` | 17,426 | 157 |
| rMD17-ethanol | MAE[kcal/mol] ↓ | 0.7139 <br><sub>R2 0.9491</sub> | ~0.009 kcal/mol (~0.4 meV) | +0.9491 | `e_norm→e_painn→e_gate` | 34,877 | 72 |
| rMD17-aspirin | MAE[kcal/mol] ↓ | 3.8499 <br><sub>R2 0.3897</sub> | ~0.05 kcal/mol (~2.2 meV) | +0.3897 | `e_gate→e_painn→e_gate` | 34,877 | 51 |
| ModelNet10 | acc | 0.3877 | ~0.93-0.95 (PointNet++/DGCNN) | +0.3196 | `e_norm→e_painn→e_gate` | 35,110 | 70 |

**`set`** -- permutation-invariant sets (particle physics)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| JetMass | R2 | 0.9994 | ~0.9+ (EFN/Deep Sets; mass is a smooth permutation-invariant set function) | +0.9994 | `norm→deepsets→deepsets→element_mlp` | 90,203 | 24 |
| JetNet | acc | 0.7349 <br><sub>ROC-AUC 0.9254</sub> | ~0.78-0.82 (JEDI-net/PELICAN, 5-class) | +0.6686 | `norm→deepsets→element_mlp` | 78,071 | 26 |
| TopTagging | acc | 0.7220 <br><sub>1/eB@eS0.3 11.95, 1/eB@eS0.5 6.032, ROC-AUC 0.7928</sub> | ~0.93 / AUC ~0.98 | +0.4410 | `norm→deepsets→deepsets` | 80,523 | 96 |

**`operator`** -- function -> function on a grid (PDE surrogates)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| Wave2D | field_R2 | 0.9996 | ~0.99 (FNO-class) | +0.9996 | `local→local→local` | 5,048,349 | 50 |
| Burgers1D | field_R2 | 0.9971 | ~0.999 (FNO) | +0.9971 | `local→local→deeponet` | 195,909 | 59 |
| Darcy2D | field_R2 | 0.8049 | ~0.999 (FNO) | +0.8049 | `local→local→deeponet` | 5,048,349 | 1000† |

**`generated_equivariant`** -- a contract GENERATED for a group the symmetry front-end discovered, rather than one of the eight built-ins (reached under `--discover extended`)

| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |
|---|---|---|---|---|---|---|---|
| JetMassLorentz | R2 | 1.0000 | n/a (synthetic demonstrator; -> ~1.0 for an adequate set regressor) | +1.0000 | EMLP (discovered group) | 7,180 | 51 |

↓ lower is better (a physical-unit error); every other metric is higher-is-better.
‡ **QM9** -- the loader regresses raw U0 TOTAL energy (z-scored, rescaled to meV), whereas the quoted ~5-15 meV literature figure is for ATOMIZATION energy -- the standard QM9 target, obtained after subtracting atomic reference energies. The two are not comparable; read the R2 instead.
† training stopped at the epoch ceiling rather than on the convergence criterion -- these numbers are budget-limited, not converged.

<!-- END:stdval -->

Reproduce (the suite is run in per-contract batches; results merge into one document):

```bash
export ILMARINEN_DATA_DIR=$PWD/ilmarinen_data
for C in 4d operator graph spatial volumetric set equivariant sequence; do
  python -m validation_runners.run_standard_validation \
    --preset opt --select sparse --sparsity_mu 0.3 --select_size variable \
    --auto_epoch val --auto_epoch_patience 10 --auto_epoch_min_epochs 10 \
    --auto_epoch_extend 100 --auto_epoch_max 1000 --auto_epoch_restore_best \
    --save_models --contracts "$C"
done
python -m validation_runners.make_results_table --insert-readme
```

## Tests

```bash
pip install -r requirements-dev.txt      # dev-only: pytest
python -m pytest tests_unit/             # full suite (~10 s on CPU)
python -m pytest tests_unit/ -m "not smoke"   # fast subset (skips the slower end-to-end fits)
```

The suite locks the public API and the physics invariants: the pricing/MDL identities, the Gibbs
readout, routing, symmetry, the LLC/RLCT guard, the effective-dimension ledger, and one end-to-end fit
per arena.

## Reports

Three LaTeX reports ship with their `.tex` sources and PDFs:

- **`ilmarinen_report`** — *A Variational and Field-Theoretic Formulation*: the theory/formulation.
- **`ilmarinen_implementation_report`** — the implementation: the pipeline, the priced ladder, the
  contract/primitive/size machinery, the symmetry front-end, the interpretability read-outs, and
  validation.
- **`ilmarinen_technical_report`** — *A Symbolic Technical Reference*: a symbolic catalogue of what the
  package computes.

## Structure

```
ilmarinen/
├── core/          the AllGraph controller + AllData, the 8 contracts, routing, dataset loaders,
│                  symmetry discovery, IB-RG flow, redundancy reduction, interpretability
├── machinery/     the priced-selection primitives: MDL pricing (contract/depth/width/mixture),
│                  the Gibbs readout, singular complexity (LLC) + functional pricing, the
│                  thermodynamic potential, response spectroscopy, the effective-dimension ledger
├── models/        primitive operations and schema realizations (incl. equivariant, neural ODE)
└── _selfcheck.py  integrity check over all modules and primitives

validation_runners/   the quick + standard validation CLIs (every flag exposed)
studies/              standalone study reproducers
tests_unit/           the unit suite
```

## Data locations (portable)

Nothing is written to hard-coded system paths. Downloads are cached under a single base dir, resolved
as `$ILMARINEN_DATA_DIR` if set, else `<os-temp>/ilmarinen_data`. For offline use, drop pre-downloaded
dataset files into `$ILMARINEN_UPLOADS_DIR` (defaults to `<base>/uploads`); loaders fall back to those.
See `ilmarinen/core/paths.py`; set `ILMARINEN_DATA_VERBOSE=1` to log data provenance.

## Citation

If you use `ilmarinen` in academic work, please cite it — see [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button exports BibTeX). A versioned DOI is minted for each
GitHub Release once the repository is enabled on Zenodo.

## License

Licensed under the [Apache License 2.0](LICENSE) — permissive use, modification, and
redistribution with an explicit patent grant. Copyright 2026 Rostislav Duda.
