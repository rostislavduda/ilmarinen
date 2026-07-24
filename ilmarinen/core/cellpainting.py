"""JUMP Cell Painting Gallery loader (via the `jump-portrait` package).

Fetches multi-channel fluorescence microscopy images of human cells under CRISPR genetic perturbation
directly from the PUBLIC Cell Painting Gallery S3 bucket (no AWS account needed), and frames a K-class
"identify the perturbation from cell morphology" classification task for the SPATIAL schema.

Each sample is one microscopy field -- a (plate, well, site) imaged in the 5 canonical Cell Painting
channels (DNA, RNA, ER, AGP, Mito) -- robustly contrast-normalized, resized to a fixed hw x hw, and stacked
into a (5, hw, hw) tensor. Restricted to a single imaging SOURCE so acquisition is consistent across classes.
The assembled arrays are cached as an npz under the ilmarinen data dir (paths.data_dir), so a run fetches from
S3 once and reuses it thereafter. Train/test are split by WELL (a well's fields never straddle the split) to
avoid field-of-view leakage.

Requires `jump-portrait` (pip install jump-portrait) and network access to the gallery S3 at first build.
"""
from __future__ import annotations
import os
import numpy as np

# 6 CRISPR genes with distinct, well-studied morphological signatures, all present on source_13 with >=10
# wells each (cell-cycle / kinase / signaling perturbations that Cell Painting separates well).
CLASSES = ("CDK1", "AURKB", "EGFR", "RAC1", "KRAS", "TP53")
CHANNELS = ("DNA", "RNA", "ER", "AGP", "Mito")
SOURCE = "source_13"


#: native pixel dimensions of the last fetched field, recorded by _robust_norm_resize. A resolution study
#: needs the NATIVE size to convert an hw into a physical sampling rate (um/pixel) and ask which structures
#: still clear Nyquist -- nuclei are coarse and survive aggressive downsampling, mitochondrial/AGP
#: granularity is fine and is destroyed first. Populated as a side effect of building the cache.
NATIVE_SHAPE = {}


def _robust_norm_resize(img, hw):
    """uint16 microscopy image -> float32 (hw,hw) in ~[0,1] via per-image 1-99 percentile contrast scaling
    (microscopy has a long bright tail; percentile clipping is the standard robust stretch), then resize."""
    from skimage.transform import resize
    a = img.astype(np.float32)
    NATIVE_SHAPE["shape"] = tuple(a.shape)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return resize(a, (hw, hw), anti_aliasing=True, preserve_range=True).astype(np.float32)


def build_cellpainting(hw=48, per_class=12, sites_per_well=2, classes=CLASSES, channels=CHANNELS, source=SOURCE):
    """Fetch and assemble the dataset from S3. Returns (X (n,C,hw,hw) float32, y (n,) int64, wells (n,) str,
    class_names). Per class, take up to `per_class` distinct WELLS and up to `sites_per_well` fields (sites)
    from each -- so the by-well train/test split stays valid AND each class has several fields per well for
    the retrieval metric."""
    from jump_portrait.fetch import get_item_location_metadata, get_jump_image_batch
    import pyarrow as pa
    import pyarrow.compute as pc
    Xs, ys, wells = [], [], []
    for ci, gene in enumerate(classes):
        meta = get_item_location_metadata(gene)                    # pyarrow Table of (source,plate,well,site,urls)
        meta = meta.filter(pc.equal(meta.column("Metadata_Source"), source))
        wells_seen, rows = {}, []
        for r in meta.to_pylist():
            wk = (r["Metadata_Plate"], r["Metadata_Well"])
            cnt = wells_seen.get(wk, 0)
            if cnt == 0 and len(wells_seen) >= per_class:
                continue                                           # already have enough distinct wells
            if cnt >= sites_per_well:
                continue                                           # already have enough fields for this well
            wells_seen[wk] = cnt + 1; rows.append(r)
        if not rows:
            continue
        sub = pa.Table.from_pylist(rows, schema=meta.schema)
        md, imgs = get_jump_image_batch(sub, channels=list(channels), site=None)   # parallel S3 fetch
        # regroup the flat (field x channel) output into one (C,hw,hw) stack per field
        groups = {}
        for d, im in zip(md, imgs):
            if im is None:
                continue
            key = (d["Metadata_Plate"], d["Metadata_Well"], d["Metadata_Site"])
            groups.setdefault(key, {})[d["Metadata_Channel"]] = im
        for (plate, well, site), chdict in groups.items():
            if not all(c in chdict for c in channels):             # need every channel present
                continue
            Xs.append(np.stack([_robust_norm_resize(chdict[c], hw) for c in channels], 0))
            ys.append(ci); wells.append(f"{plate}/{well}")
    if not Xs:
        raise RuntimeError("Cell Painting: no fields fetched (check jump-portrait / S3 access).")
    return np.stack(Xs, 0).astype(np.float32), np.asarray(ys, np.int64), np.asarray(wells), list(classes)


def _cached_arrays(hw, per_class, sites_per_well, classes):
    """Build once, cache to an npz under the ilmarinen data dir; reuse on later runs (keyed by the knobs)."""
    from .paths import data_dir
    tag = f"cellpainting_{'_'.join(classes)}_s{SOURCE}_hw{hw}_pc{per_class}_spw{sites_per_well}.npz"
    path = os.path.join(data_dir(), tag)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], d["y"], d["wells"], list(d["classes"])
    X, y, wells, cls = build_cellpainting(hw=hw, per_class=per_class, sites_per_well=sites_per_well, classes=classes)
    np.savez(path, X=X, y=y, wells=wells, classes=np.array(cls))
    return X, y, wells, cls


def resample_stack(X, hw):
    """Resize a cached (n, C, H, W) field stack to (n, C, hw, hw) with anti-aliasing.

    Lets a RESOLUTION study derive every lower hw from ONE high-resolution build instead of re-fetching
    the whole gallery from S3 per hw (the cache key in _cached_arrays includes hw, so a naive sweep pays
    the S3 cost N times). The per-image percentile stretch in _robust_norm_resize already happened at
    native resolution, so this is a pure resampling step. NOTE it is a two-stage resample
    (native -> build_hw -> hw), which is not bit-identical to a direct native -> hw resize; it is the
    controlled choice for a sweep (identical fields and normalization, resolution the only variable),
    but confirm a CHOSEN hw against a direct build before shipping it."""
    from skimage.transform import resize
    X = np.asarray(X)
    if X.shape[-1] == hw and X.shape[-2] == hw:
        return X.astype(np.float32)
    n, c = X.shape[0], X.shape[1]
    out = np.empty((n, c, hw, hw), np.float32)
    for i in range(n):
        for j in range(c):
            out[i, j] = resize(X[i, j], (hw, hw), anti_aliasing=True, preserve_range=True)
    return out


def split_by_well(X, y, wells, classes, split_seed=0):
    """Per-channel z-score + the stratified BY-WELL train/test split, as the standard train/test/task/...
    dict. Split out of load_cellpainting so a resolution/robustness study can re-split the SAME arrays
    under different seeds -- the effective sample size here is WELLS, not fields (a well's sites are near
    replicates), so honest error bars come from resampling this split, not from resampling fields."""
    from .allgraph import AllData
    import torch
    # per-channel z-score (channels are distinct stains with different dynamic ranges)
    mu = X.mean((0, 2, 3), keepdims=True); sd = X.std((0, 2, 3), keepdims=True) + 1e-6
    X = (X - mu) / sd
    # split by WELL, STRATIFIED per class -- a well's fields never straddle train/test (no field-of-view
    # leakage) and every class contributes wells to both sides (so the retrieval metric is well-posed).
    rng = np.random.RandomState(split_seed)
    by_class = {}
    for w, lab in zip(wells.tolist(), y.tolist()):
        by_class.setdefault(int(lab), set()).add(w)
    te_wells = set()
    for lab, ws in by_class.items():
        ws = sorted(ws); rng.shuffle(ws)
        k = max(1, int(round(0.25 * len(ws)))) if len(ws) > 1 else 0
        te_wells.update(ws[:k])
    te_mask = np.array([w in te_wells for w in wells]); tr_mask = ~te_mask
    Xt = torch.tensor(X)
    return {"train": AllData.dense_tensor(Xt[tr_mask], y=y[tr_mask]),
            "test": AllData.dense_tensor(Xt[te_mask], y=y[te_mask]),
            "task": "classification", "chance": 1.0 / len(classes), "class_names": list(classes),
            "field": "cell biology (JUMP Cell Painting, CRISPR perturbation ID)",
            "sota": "acc n/a (custom 6-gene perturbation-ID task; CP benchmarks report retrieval mAP)"}


def load_cellpainting(reduced=False, device="cpu", per_class=None, sites_per_well=2, hw=48, classes=None,
                      split_seed=0):
    """JUMP Cell Painting K-class perturbation-ID for the spatial contract. `per_class` (distinct wells per
    class) is the data-amount knob; when None it defaults from `reduced` (6 quick / 18 full). Returns the
    standard train/test/task/... dict plus 'class_names' (for the retrieval metric)."""
    if per_class is None:
        per_class = 6 if reduced else 18
    classes = tuple(classes) if classes else CLASSES
    X, y, wells, classes = _cached_arrays(hw, per_class, sites_per_well, classes)
    return split_by_well(X, y, wells, classes, split_seed=split_seed)
