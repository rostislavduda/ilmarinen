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

import contextlib
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


@contextlib.contextmanager
def _image_bar(total, enabled=True):
    """A single live tqdm bar over the TOTAL images to download, in the same style as the training loop's
    epoch bar (leave=False, dynamic width). Yields the tqdm object, or None when progress is off, tqdm is
    unavailable, or there is nothing to fetch -- callers must treat None as 'no bar'."""
    if not (enabled and total and total > 0):
        yield None
        return
    try:
        from tqdm.auto import tqdm
    except Exception:
        yield None
        return
    bar = tqdm(total=total, desc="  importing Cell Painting", unit="img", leave=False, dynamic_ncols=True)
    try:
        yield bar
    finally:
        bar.close()


@contextlib.contextmanager
def _tqdm_joblib(bar, n_jobs=8):
    """Advance `bar` once per IMAGE across jump_portrait's S3 fetch inside get_jump_image_batch (a call we
    don't own). TWO things are needed, and they are inseparable:

      1. get_jump_image_batch runs a bare `joblib.Parallel()`, which defaults to the SEQUENTIAL backend --
         and joblib's sequential path never invokes a batch-completion callback, so a callback patch alone
         does nothing (the bar just sits at 0). We therefore open an enclosing threading backend with
         n_jobs>1, which the bare Parallel() inherits; that both makes the callback fire AND parallelizes
         the I/O-bound S3 GETs (the library default is sequential, so this is also a fetch speedup).
      2. we subclass joblib's per-batch completion callback to update the bar by the number of tasks it
         reports done (== images), so with real (slow) downloads the bar advances smoothly, one per image.

    A no-op when bar is None. If joblib's internals differ from what we patch, it degrades gracefully -- the
    fetch still runs (sequentially, as before) and the bar simply won't advance; it never raises."""
    if bar is None:
        yield
        return
    try:
        import joblib
        base = joblib.parallel.BatchCompletionCallBack
        # parallel_config (joblib>=1.3) or the older parallel_backend -- either forces a callback-firing
        # backend; both take (backend, n_jobs=...). Needed because n_jobs=1 fires no completion callback.
        force_backend = getattr(joblib, "parallel_config", None) or joblib.parallel_backend
    except Exception:
        yield
        return

    class _Cb(base):
        def __call__(self, *a, **k):
            try:
                bar.update(self.batch_size)          # batch_size tasks (== images) completed in this callback
            except Exception:
                pass
            return super().__call__(*a, **k)

    joblib.parallel.BatchCompletionCallBack = _Cb
    try:
        with force_backend("threading", n_jobs=n_jobs):
            yield
    finally:
        joblib.parallel.BatchCompletionCallBack = base


def _robust_norm_resize(img, hw):
    """uint16 microscopy image -> float32 (hw,hw) in ~[0,1] via per-image 1-99 percentile contrast scaling
    (microscopy has a long bright tail; percentile clipping is the standard robust stretch), then resize."""
    from skimage.transform import resize
    a = img.astype(np.float32)
    NATIVE_SHAPE["shape"] = tuple(a.shape)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return resize(a, (hw, hw), anti_aliasing=True, preserve_range=True).astype(np.float32)


def _select_rows(rows, per_class, sites_per_well):
    """Deterministic greedy pick of the CANONICAL selection for ONE class: scan `rows` (metadata dicts, in
    index order) and take up to `per_class` distinct wells, up to `sites_per_well` sites each. Returns
    [(row, well_rank, site_rank), ...] where well_rank (0-based, first-seen order among all scanned rows)
    and site_rank (0-based within its well, accept order) are STABLE canonical coordinates: because the scan
    order is fixed and a well's rank is just its first-appearance index (independent of the quotas), the
    predicate well_rank<N & site_rank<M reproduces exactly the selection for (per_class=N, sites_per_well=M).
    That stability is what lets the on-disk cache grow monotonically -- a larger request is a strict superset
    of a smaller one, so we only ever fetch the delta."""
    wrank = {}                                   # well_key -> its well_rank (assigned once, on first sight)
    scount = {}                                  # well_key -> next free site_rank
    out = []
    for r in rows:
        wk = (r["Metadata_Plate"], r["Metadata_Well"])
        if wk not in wrank:
            if len(wrank) >= per_class:
                continue                         # this class already has its full quota of distinct wells
            wrank[wk] = len(wrank); scount[wk] = 0
        if scount[wk] >= sites_per_well:
            continue                             # this well already has its full quota of sites
        out.append((r, wrank[wk], scount[wk]))
        scount[wk] += 1
    return out


def _subset_indices(y, wrank, srank, per_class, sites_per_well):
    """Indices of the cached fields that constitute the request (per_class wells x sites_per_well sites per
    class): every field whose canonical well_rank < per_class AND site_rank < sites_per_well. Purely a
    function of the stored ranks -- independent of cache array order and needing no network -- sorted by
    (class, well_rank, site_rank) for a stable, reproducible ordering."""
    keep = [i for i in range(len(y)) if wrank[i] < per_class and srank[i] < sites_per_well]
    keep.sort(key=lambda i: (int(y[i]), int(wrank[i]), int(srank[i])))
    return np.asarray(keep, dtype=np.int64)


def _plan_class(gene, source, per_class, sites_per_well, have):
    """Metadata-only planning for one class (NO downloads): query the gene's field locations, restrict to
    `source`, canonically select up to (per_class wells, sites_per_well sites), and return (to_fetch_table,
    rank_map). rank_map maps (plate, well, str(site)) -> (well_rank, site_rank) for EVERY selected field;
    to_fetch_table is the pyarrow sub-table of only the fields NOT already in `have` (a set of identities
    (f"{plate}/{well}", str(site))), or None when the class needs no new download."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from jump_portrait.fetch import get_item_location_metadata
    meta = get_item_location_metadata(gene)                        # pyarrow Table of (source,plate,well,site,urls)
    meta = meta.filter(pc.equal(meta.column("Metadata_Source"), source))
    rmap, to_fetch = {}, []
    for r, wr, sr in _select_rows(meta.to_pylist(), per_class, sites_per_well):
        plate, well, site = r["Metadata_Plate"], r["Metadata_Well"], str(r["Metadata_Site"])
        rmap[(plate, well, site)] = (wr, sr)
        if (f"{plate}/{well}", site) not in have:
            to_fetch.append(r)
    sub = pa.Table.from_pylist(to_fetch, schema=meta.schema) if to_fetch else None
    return sub, rmap


def _fetch_rows(plans, channels, hw, progress=True, fetch_jobs=8):
    """Download + assemble the planned fields. `plans` is a list of (ci, sub_table, rank_map). Returns the
    parallel arrays (X, y, wells, sites, wrank, srank) for the fields that came back with ALL channels, or
    None if nothing was fetched. One shared tqdm bar advances per IMAGE across every class (see _tqdm_joblib
    for why a threading backend is forced). Each assembled field carries its canonical ranks from rank_map."""
    from jump_portrait.fetch import get_jump_image_batch
    total_imgs = sum(sub.num_rows * len(channels) for _, sub, _ in plans)
    Xs, ys, wells, sites, wr, sr = [], [], [], [], [], []
    with _image_bar(total_imgs, enabled=progress) as bar:
        for ci, sub, rmap in plans:
            with _tqdm_joblib(bar, n_jobs=fetch_jobs):
                md, imgs = get_jump_image_batch(sub, channels=list(channels), site=None)   # parallel S3 fetch
            # regroup the flat (field x channel) output into one (C,hw,hw) stack per field
            groups = {}
            for d, im in zip(md, imgs):
                if im is None:
                    continue
                key = (d["Metadata_Plate"], d["Metadata_Well"], str(d["Metadata_Site"]))
                groups.setdefault(key, {})[d["Metadata_Channel"]] = im
            for key, chdict in groups.items():
                if key not in rmap:                                # defensive: only assemble planned fields
                    continue
                if not all(c in chdict for c in channels):         # need every channel present
                    continue
                plate, well, site = key
                Xs.append(np.stack([_robust_norm_resize(chdict[c], hw) for c in channels], 0))
                w, s = rmap[key]
                ys.append(ci); wells.append(f"{plate}/{well}"); sites.append(site); wr.append(w); sr.append(s)
    if not Xs:
        return None
    return (np.stack(Xs, 0).astype(np.float32), np.asarray(ys, np.int64), np.asarray(wells),
            np.asarray(sites), np.asarray(wr, np.int64), np.asarray(sr, np.int64))


def build_cellpainting(hw=48, per_class=12, sites_per_well=2, classes=CLASSES, channels=CHANNELS, source=SOURCE,
                       progress=True, fetch_jobs=8):
    """Fetch and assemble the dataset from S3, FROM SCRATCH. Returns (X (n,C,hw,hw) float32, y (n,) int64,
    wells (n,) str, class_names). Per class, take up to `per_class` distinct WELLS and up to `sites_per_well`
    fields (sites) each. Normal callers go through the incrementally-cached `_cached_arrays`; this is the
    no-cache builder it (and any direct caller) is built on. `progress` shows a per-IMAGE tqdm bar."""
    plans = []
    for ci, gene in enumerate(classes):
        sub, rmap = _plan_class(gene, source, per_class, sites_per_well, have=set())   # have=empty -> fetch all
        if sub is not None:
            plans.append((ci, sub, rmap))
    got = _fetch_rows(plans, channels, hw, progress=progress, fetch_jobs=fetch_jobs) if plans else None
    if got is None:
        raise RuntimeError("Cell Painting: no fields fetched (check jump-portrait / S3 access).")
    X, y, wells, sites, wr, sr = got
    return X, y, wells, list(classes)


def _cached_arrays(hw, per_class, sites_per_well, classes, progress=True, fetch_jobs=8):
    """Serve the (per_class wells x sites_per_well sites) request for `classes`, backed by a SINGLE
    incrementally-grown cache per (classes, source, hw) -- note the cache key no longer includes per_class /
    sites_per_well, so scaling the amount up reuses everything already on disk.

    The cache stores every field ever fetched plus its canonical (well_rank, site_rank) identity and a
    high-water mark (pc_max, spw_max) = the largest amount ever requested. Two paths:
      * FAST  (per_class <= pc_max and sites_per_well <= spw_max): subset the cache by rank and return with
        NO network and NO jump_portrait import -- an instant np.load, just like before.
      * GROW  (otherwise): fetch ONLY the fields the larger request adds (never re-downloading what is on
        disk -- see _plan_class's `have` filter), append them, bump the high-water mark to the new max in
        each dimension, and re-save. A per-image bar covers just the delta.
    Returns the request's (X, y, wells, classes) for split_by_well.

    NOTE: the cache-file naming changed from the older `..._pc{n}_spw{m}.npz` scheme, so a pre-existing
    legacy cache is not reused (it is simply ignored, and the first run rebuilds under the new name)."""
    from .paths import data_dir
    tag = f"cellpainting_{'_'.join(classes)}_s{SOURCE}_hw{hw}.npz"
    path = os.path.join(data_dir(), tag)

    # --- load whatever is already cached (fields + canonical ranks + high-water mark) ---
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        X, y, wells, sites = d["X"], d["y"], d["wells"], d["sites"]
        wrank, srank = d["wrank"], d["srank"]
        pc_max, spw_max = int(d["pc_max"]), int(d["spw_max"])
    else:
        X = np.empty((0, len(CHANNELS), hw, hw), np.float32)
        y = np.empty((0,), np.int64)
        wells = np.empty((0,), dtype="<U1"); sites = np.empty((0,), dtype="<U1")
        wrank = np.empty((0,), np.int64); srank = np.empty((0,), np.int64)
        pc_max = spw_max = 0

    # --- FAST PATH: the request already fits within the high-water mark -> rank-subset, no network ---
    if per_class <= pc_max and sites_per_well <= spw_max:
        idx = _subset_indices(y, wrank, srank, per_class, sites_per_well)
        return X[idx], y[idx], wells[idx], list(classes)

    # --- GROW PATH: fetch only the delta up to the new high-water mark (per-dimension max) ---
    target_pc, target_spw = max(per_class, pc_max), max(sites_per_well, spw_max)
    have = set(zip(wells.tolist(), sites.tolist()))                # identities already on disk
    plans = []
    for ci, gene in enumerate(classes):
        sub, rmap = _plan_class(gene, SOURCE, target_pc, target_spw, have)
        if sub is not None:
            plans.append((ci, sub, rmap))
    got = _fetch_rows(plans, CHANNELS, hw, progress=progress, fetch_jobs=fetch_jobs) if plans else None
    if got is not None:
        nX, ny, nwells, nsites, nwr, nsr = got
        X = np.concatenate([X, nX]); y = np.concatenate([y, ny])
        wells = np.concatenate([wells, nwells]); sites = np.concatenate([sites, nsites])
        wrank = np.concatenate([wrank, nwr]); srank = np.concatenate([srank, nsr])
    pc_max, spw_max = target_pc, target_spw                        # remember we've now covered this amount
    if len(X) == 0:
        raise RuntimeError("Cell Painting: no fields fetched (check jump-portrait / S3 access).")
    np.savez(path, X=X, y=y, wells=wells, sites=sites, wrank=wrank, srank=srank,
             classes=np.array(list(classes)), pc_max=pc_max, spw_max=spw_max)
    idx = _subset_indices(y, wrank, srank, per_class, sites_per_well)
    return X[idx], y[idx], wells[idx], list(classes)


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
    import torch

    from .allgraph import AllData
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
                      split_seed=0, progress=True, fetch_jobs=8):
    """JUMP Cell Painting K-class perturbation-ID for the spatial contract. `per_class` (distinct wells per
    class) is the data-amount knob; when None it defaults from `reduced` (6 quick / 18 full). Returns the
    standard train/test/task/... dict plus 'class_names' (for the retrieval metric).

    The dataset grows ON DEMAND: the S3 fetch is backed by a single incrementally-extended cache (see
    _cached_arrays), so raising `per_class`/`sites_per_well` across runs downloads ONLY the additional
    wells/sites and reuses everything already on disk; lowering them is served instantly from cache. `progress`
    shows the per-image tqdm bar over whatever delta is being fetched; `fetch_jobs` sets S3 fetch concurrency."""
    if per_class is None:
        per_class = 6 if reduced else 18
    classes = tuple(classes) if classes else CLASSES
    X, y, wells, classes = _cached_arrays(hw, per_class, sites_per_well, classes, progress=progress,
                                          fetch_jobs=fetch_jobs)
    return split_by_well(X, y, wells, classes, split_seed=split_seed)
