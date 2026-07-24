# Ilmarinen — a physics-principled neural-architecture meta-optimizer

[![CI](https://github.com/rostislavduda/ilmarinen/actions/workflows/ci.yml/badge.svg)](https://github.com/rostislavduda/ilmarinen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rostislavduda/ilmarinen/branch/main/graph/badge.svg)](https://codecov.io/gh/rostislavduda/ilmarinen)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

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
