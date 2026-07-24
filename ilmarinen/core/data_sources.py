"""Portable data acquisition: fetch each benchmark dataset from its canonical Python package / public URL,
falling back to a locally-uploaded copy if the fetch fails.

Motivation: the loaders originally read fixed absolute paths, which only work in the environment where those
files were placed. To let the validation suites run from ANY machine, every dataset is fetched from its standard
source here; an optional local "uploads" directory is kept only as an offline fallback (some public mirrors
-- Zenodo, figshare, UCI-static -- may be blocked on some networks, in which case the fallback is used, while the
package/URL path runs everywhere else).

Each helper returns the same in-memory structure the loaders expect and never hard-fails as long as EITHER the
network source OR the uploaded fallback is available. A single cache dir (see ilmarinen.core.paths.data_dir --
overridable via $ILMARINEN_DATA_DIR, defaults to <os-temp>/ilmarinen_data) holds anything downloaded so repeated
runs don't re-fetch. Offline dataset files can be dropped in $ILMARINEN_UPLOADS_DIR (see paths.uploads_dir).
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np

from .paths import data_dir, uploads_dir

CACHE = data_dir()
UPLOADS = uploads_dir()


def _log(msg):
    # lightweight, only when the caller wants to see acquisition provenance
    if os.environ.get("ILMARINEN_DATA_VERBOSE"):
        print(f"[data_sources] {msg}")


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _http_get_to_file(url, dst, timeout=180):
    """Stream a (possibly large) URL to `dst` via urllib, writing through a .part temp so an interrupted
    download never leaves a truncated file in the cache."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    return dst


def _requests_get_to_file(url, dst, timeout=300):
    """Stream a large URL to `dst` via the `requests` package. Some public mirrors (notably Zenodo, served
    behind Cloudflare) reject urllib's client signature with HTTP 403 but serve `requests`/browsers fine, so
    downloads from those hosts go through here."""
    import requests  # standard, ships with the scientific stack; declared in requirements
    tmp = dst + ".part"
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"},
                      stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, dst)
    return dst


# --------------------------------------------------------------------------- MedMNIST (2D and 3D)
def medmnist_arrays(flag, size=28):
    """Return dict(train/val/test _images, _labels) for a MedMNIST dataset `flag` (e.g. 'bloodmnist',
    'organmnist3d'). Tries the medmnist package (auto-download), then a locally-uploaded <flag>.npz.

    The medmnist npz layout is exactly {split_images, split_labels}, so the uploaded files and the package
    output are interchangeable."""
    # 1) package path (works on any networked machine)
    try:
        import medmnist
        from medmnist import INFO
        cls = getattr(medmnist, INFO[flag]["python_class"])
        out = {}
        for split in ("train", "val", "test"):
            try:
                ds = cls(split=split, download=True, root=CACHE, size=size)
            except TypeError:  # older medmnist without size=
                ds = cls(split=split, download=True, root=CACHE)
            out[f"{split}_images"] = ds.imgs
            out[f"{split}_labels"] = ds.labels
        _log(f"{flag}: loaded via medmnist package")
        return out
    except Exception as e:
        _log(f"{flag}: medmnist package path failed ({str(e)[:50]}); trying uploaded npz")

    # 2) uploaded fallback
    for cand in (os.path.join(UPLOADS, f"{flag}.npz"), os.path.join(CACHE, f"{flag}.npz")):
        if os.path.exists(cand):
            d = np.load(cand)
            _log(f"{flag}: loaded uploaded fallback {cand}")
            return {k: d[k] for k in d.files}
    raise RuntimeError(f"{flag}: could not fetch via medmnist and no uploaded {flag}.npz found. Install "
                       f"medmnist with network access, or place {flag}.npz in {UPLOADS}.")


# --------------------------------------------------------------------------- rMD17 molecules
# The Revised MD17 figshare record now publishes per-molecule .npz files directly (alongside the ~1 GB
# rmd17.tar.bz2 archive), so each molecule is fetched on its own without pulling the whole dataset.
_RMD17_FIGSHARE_ARTICLE = 12672038      # figshare doi:10.6084/m9.figshare.12672038 (Revised MD17)


def _figshare_rmd17_url(molecule):
    """Resolve the direct download URL for rmd17_<molecule>.npz from the figshare article metadata (so we
    never hard-code file ids, which figshare rotates when files are revised)."""
    import json
    fname = f"rmd17_{molecule}.npz"
    meta = json.loads(_http_get(f"https://api.figshare.com/v2/articles/{_RMD17_FIGSHARE_ARTICLE}", timeout=60))
    for f in meta.get("files", []):
        if f.get("name") == fname:
            return f["download_url"]
    raise RuntimeError(f"figshare article {_RMD17_FIGSHARE_ARTICLE} lists no {fname}")


def rmd17_npz(molecule):
    """Return the loaded rMD17 npz for a molecule (e.g. 'ethanol', 'aspirin'). Tries $RMD17_DIR/rmd17_<m>.npz,
    then the uploaded rmd17_<m>.npz, then the cache; if none is present, downloads the per-molecule .npz from
    the figshare record and caches it. Set RMD17_DIR to a local rMD17 folder to skip the download entirely."""
    fname = f"rmd17_{molecule}.npz"
    cands = []
    if os.environ.get("RMD17_DIR"):
        cands.append(os.path.join(os.environ["RMD17_DIR"], fname))
    cands += [os.path.join(UPLOADS, fname), os.path.join(CACHE, fname)]
    for c in cands:
        if os.path.exists(c):
            _log(f"rmd17 {molecule}: loaded {c}")
            return np.load(c)
    # fetch the per-molecule npz from figshare (works with urllib -- figshare is not Cloudflare-gated)
    dst = os.path.join(CACHE, fname)
    try:
        url = _figshare_rmd17_url(molecule)
        _log(f"rmd17 {molecule}: downloading {fname} from figshare")
        _http_get_to_file(url, dst, timeout=300)
        return np.load(dst)
    except Exception as e:
        raise RuntimeError(f"rMD17 {molecule}: no {fname} found and figshare download failed ({str(e)[:60]}). "
                           f"Download rMD17 (doi:10.6084/m9.figshare.{_RMD17_FIGSHARE_ARTICLE}) and set "
                           f"RMD17_DIR, or place {fname} in {UPLOADS}.")


# --------------------------------------------------------------------------- QM7
def qm7_mat_path():
    """Return a filesystem path to qm7.mat. Tries the public URL (quantum-machine.org), then the uploaded copy.
    The .mat is small (~15 MB)."""
    cached = os.path.join(CACHE, "qm7.mat")
    if os.path.exists(cached):
        return cached
    # 1) public URL
    for url in ("http://quantum-machine.org/data/qm7.mat",
                "https://ndownloader.figshare.com/files/24895834"):
        try:
            data = _http_get(url, timeout=60)
            with open(cached, "wb") as f:
                f.write(data)
            _log(f"qm7: downloaded from {url}")
            return cached
        except Exception as e:
            _log(f"qm7: URL {url} failed ({str(e)[:40]})")
    # 2) uploaded fallback
    up = os.path.join(UPLOADS, "qm7.mat")
    if os.path.exists(up):
        _log("qm7: using uploaded qm7.mat")
        return up
    raise RuntimeError("qm7.mat not found via URL or upload; place qm7.mat in " + UPLOADS)


# --------------------------------------------------------------------------- QM9
def qm9_xyz_dir(n_files):
    """Return a directory containing at least `n_files` QM9 .xyz files (the dsgdb9nsd.xyz inner dir). Tries an
    already-extracted cache, then the uploaded qm9.zip, then the public figshare tarball. Extracts only the
    first n_files to stay compact."""
    inner = os.path.join(CACHE, "qm9", "dsgdb9nsd.xyz")
    have = os.path.isdir(inner) and len([f for f in os.listdir(inner) if f.endswith(".xyz")]) >= n_files
    if have:
        return inner

    def _extract_from_zip(zpath):
        os.makedirs(os.path.join(CACHE, "qm9"), exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            names = [f for f in z.namelist() if f.endswith(".xyz")][:n_files]
            for nm in names:
                z.extract(nm, os.path.join(CACHE, "qm9"))
        return inner if os.path.isdir(inner) else os.path.join(CACHE, "qm9")

    # 1) uploaded qm9.zip (primary offline source; the full QM9 download is large)
    up = os.path.join(UPLOADS, "qm9.zip")
    if os.path.exists(up):
        _log("qm9: extracting subset from uploaded qm9.zip")
        return _extract_from_zip(up)
    # 2) public tarball (figshare) -- large; only used if no upload
    try:
        url = "https://ndownloader.figshare.com/files/3195389"    # dsgdb9nsd.xyz.tar.bz2
        data = _http_get(url, timeout=180)
        tarpath = os.path.join(CACHE, "qm9.tar.bz2")
        with open(tarpath, "wb") as f:
            f.write(data)
        import tarfile
        os.makedirs(os.path.join(CACHE, "qm9"), exist_ok=True)
        with tarfile.open(tarpath, "r:bz2") as t:
            members = [m for m in t.getmembers() if m.name.endswith(".xyz")][:n_files]
            t.extractall(os.path.join(CACHE, "qm9"), members=members)
        _log("qm9: downloaded+extracted subset from figshare")
        return inner if os.path.isdir(inner) else os.path.join(CACHE, "qm9")
    except Exception as e:
        raise RuntimeError(f"QM9 not available: no uploaded qm9.zip and figshare fetch failed ({str(e)[:40]}). "
                           f"Place qm9.zip in {UPLOADS} or ensure network access to figshare.")


# --------------------------------------------------------------------------- superconductivity (UCI)
def superconductivity_arrays():
    """Return (X, y) for UCI Superconductivity. Tries the ucimlrepo package, then the public UCI zip URL, then
    the uploaded superconductivty_data.zip. X = 81 features, y = critical_temp."""
    # 1) ucimlrepo package
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=464)
        X = ds.data.features.to_numpy(dtype=np.float32)
        y = ds.data.targets.to_numpy(dtype=np.float32).ravel()
        _log("superconductivity: loaded via ucimlrepo")
        return X, y
    except Exception as e:
        _log(f"superconductivity: ucimlrepo failed ({str(e)[:40]}); trying URL/upload")

    # 2) public UCI zip URL, then uploaded fallback
    import csv
    zbytes = None
    try:
        zbytes = _http_get("https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip", timeout=90)
    except Exception:
        up = os.path.join(UPLOADS, "superconductivty_data.zip")
        if os.path.exists(up):
            zbytes = open(up, "rb").read()
    if zbytes is None:
        raise RuntimeError("superconductivity: no ucimlrepo, URL blocked, and no uploaded zip in " + UPLOADS)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        with z.open("train.csv") as f:
            rows = list(csv.reader(io.TextIOWrapper(f)))
    arr = np.array([[float(v) for v in r] for r in rows[1:]], dtype=np.float32)
    return arr[:, :-1], arr[:, -1]


# --------------------------------------------------------------------------- MoleculeNet CSVs (ESOL, Tox21)
_MOLNET_URLS = {
    "delaney-processed.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "tox21.csv.gz": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}


def moleculenet_csv(fname):
    """Return a filesystem path to a MoleculeNet CSV (e.g. 'delaney-processed.csv' for ESOL, 'tox21.csv.gz').
    Tries the public DeepChem S3 URL, then the local cache, then the uploaded copy. Small files."""
    for cand in (os.path.join(CACHE, fname), os.path.join(UPLOADS, fname)):
        if os.path.exists(cand):
            return cand
    url = _MOLNET_URLS[fname]
    dst = os.path.join(CACHE, fname)
    data = _http_get(url, timeout=60)
    with open(dst, "wb") as f:
        f.write(data)
    _log(f"{fname}: downloaded from DeepChem S3")
    return dst


# --------------------------------------------------------------------------- ECG5000 (UCR)
def ecg5000_dir():
    """Return a directory holding ECG5000_TRAIN.txt / ECG5000_TEST.txt. Tries the local cache, the uploaded
    ECG5000.zip, then the public timeseriesclassification.com UCR zip."""
    cached = os.path.join(CACHE, "ecg5000")
    if os.path.exists(os.path.join(cached, "ECG5000_TRAIN.txt")):
        return cached
    out = os.path.join(CACHE, "ecg5000")
    os.makedirs(out, exist_ok=True)

    def _unzip(zbytes):
        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            for nm in z.namelist():
                if nm.endswith(("ECG5000_TRAIN.txt", "ECG5000_TEST.txt")):
                    with z.open(nm) as src, open(os.path.join(out, os.path.basename(nm)), "wb") as dst:
                        dst.write(src.read())
        return out
    # uploaded zip first (present in the sandbox)
    up = os.path.join(UPLOADS, "ECG5000.zip")
    if os.path.exists(up):
        _log("ecg5000: extracting uploaded ECG5000.zip")
        return _unzip(open(up, "rb").read())
    # public UCR mirror
    try:
        zbytes = _http_get("https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip", timeout=60)
        _log("ecg5000: downloaded from timeseriesclassification.com")
        return _unzip(zbytes)
    except Exception as e:
        raise RuntimeError(f"ECG5000 not available ({str(e)[:40]}); place ECG5000.zip in {UPLOADS}.")


# --------------------------------------------------------------------------- JetNet (Zenodo)
_JETNET_ZENODO_RECORD = "6975118"       # JetNet, 30 particles/jet, classes g/q/t/w/z (Kansal et al. 2021)


def jetnet_hdf5_dir(classes=("g", "q", "t", "w", "z")):
    """Return a directory holding <class>.hdf5 for each JetNet jet class. Uses an uploaded copy if one holds
    every class, else downloads the missing files from the JetNet Zenodo record into the cache. Zenodo sits
    behind Cloudflare (which 403s urllib), so the download goes through `requests` -- the same source the
    `jetnet` PyPI package pulls from, without its heavy, numpy-pinning dependency tree."""
    # a directory that already has every requested class (uploaded offline copy, or a completed cache)
    for base in (UPLOADS, os.path.join(CACHE, "jetnet")):
        if all(os.path.exists(os.path.join(base, f"{c}.hdf5")) for c in classes):
            _log(f"jetnet: using {base}")
            return base
    out = os.path.join(CACHE, "jetnet")
    os.makedirs(out, exist_ok=True)
    for c in classes:
        dst = os.path.join(out, f"{c}.hdf5")
        if os.path.exists(dst):
            continue
        url = f"https://zenodo.org/api/records/{_JETNET_ZENODO_RECORD}/files/{c}.hdf5/content"
        _log(f"jetnet: downloading {c}.hdf5 from Zenodo")
        try:
            _requests_get_to_file(url, dst, timeout=600)
        except Exception as e:
            raise RuntimeError(f"JetNet {c}.hdf5: Zenodo download failed ({str(e)[:60]}) and no complete "
                               f"uploaded copy in {UPLOADS}. Place g/q/t/w/z.hdf5 there for offline use.")
    return out


def pyg_temporal_json(name):
    """Return the parsed JSON for a PyTorch-Geometric-Temporal dataset (e.g. 'chickenpox', 'england_covid').
    Fetched from the public GitHub mirror (works in the sandbox); cached locally."""
    import json
    cached = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(cached):
        return json.load(open(cached))
    url = (f"https://raw.githubusercontent.com/benedekrozemberczki/"
           f"pytorch_geometric_temporal/master/dataset/{name}.json")
    data = _http_get(url, timeout=40)
    with open(cached, "wb") as f:
        f.write(data)
    _log(f"{name}: fetched from PyG-Temporal GitHub mirror")
    return json.loads(data)


# --------------------------------------------------------------------------- TU Dortmund graph-kernel datasets
def tudataset_dir(name):
    """Resolve the extracted TU Dortmund graph-kernel dataset folder for `name` (e.g. 'IMDB-BINARY'). Tries
    the local cache, then an uploaded <name>.zip, then downloads the small (<1 MB for IMDB-BINARY) archive
    from the TU repository. Returns the folder holding <name>_A.txt / _graph_indicator.txt / _graph_labels.txt.
    Raises FileNotFoundError (so the runner skips) if no copy is present and the download fails."""
    inner = os.path.join(CACHE, "tudataset", name)
    if os.path.isdir(inner) and os.path.exists(os.path.join(inner, f"{name}_A.txt")):
        return inner
    os.makedirs(os.path.join(CACHE, "tudataset"), exist_ok=True)

    def _extract(zpath):
        with zipfile.ZipFile(zpath) as z:                 # TU zips extract to a top-level <name>/ folder
            z.extractall(os.path.join(CACHE, "tudataset"))
        if not os.path.exists(os.path.join(inner, f"{name}_A.txt")):
            raise FileNotFoundError(f"TUDataset {name}: archive did not contain {name}_A.txt")
        return inner

    up = os.path.join(UPLOADS, f"{name}.zip")
    if os.path.exists(up):
        return _extract(up)
    dst = os.path.join(CACHE, "tudataset", f"{name}.zip")
    try:
        _http_get_to_file(f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip", dst)
        _log(f"tudataset {name}: downloaded from TU Dortmund repository")
        return _extract(dst)
    except Exception as e:
        raise FileNotFoundError(f"TUDataset {name}: no cached/uploaded copy and download failed "
                                f"({str(e)[:60]}). Place {name}.zip in {UPLOADS}.")


# --------------------------------------------------------------------------- ModelNet10 (Princeton 3DShapeNets)
def modelnet10_dir():
    """Resolve the extracted ModelNet10 folder (Princeton 3DShapeNets CAD meshes). Tries the cache, then an
    uploaded ModelNet10.zip, then downloads the (~450 MB) archive. Returns the folder holding the 10 class
    subdirectories. Raises FileNotFoundError (so the runner skips) if unavailable and the download fails."""
    inner = os.path.join(CACHE, "ModelNet10")

    def _resolve(base):                                   # some archives nest ModelNet10/ModelNet10/<class>
        for cand in (base, os.path.join(base, "ModelNet10")):
            if os.path.isdir(os.path.join(cand, "chair")):
                return cand
        return None

    hit = _resolve(inner)
    if hit:
        return hit

    def _extract(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(CACHE)
        r = _resolve(inner)
        if r is None:
            raise FileNotFoundError("ModelNet10: archive did not contain the expected class folders")
        return r

    up = os.path.join(UPLOADS, "ModelNet10.zip")
    if os.path.exists(up):
        return _extract(up)
    dst = os.path.join(CACHE, "ModelNet10.zip")
    try:
        _http_get_to_file("http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
                          dst, timeout=600)
        _log("ModelNet10: downloaded from Princeton 3DShapeNets")
        return _extract(dst)
    except Exception as e:
        raise FileNotFoundError(f"ModelNet10: no cached/uploaded copy and download failed ({str(e)[:60]}). "
                                f"Place ModelNet10.zip in {UPLOADS}.")


# --------------------------------------------------------------------------- Top Quark Tagging (Zenodo 2603256)
def top_tagging_h5():
    """Resolve a Top Quark Tagging Reference Dataset HDF5 (Kasieczka, Plehn, Thompson & Russell; Zenodo record
    2603256). Prefers an uploaded/cached file (top_tagging.h5 / test.h5 / val.h5 / train.h5), else downloads
    test.h5 from Zenodo (Cloudflare-gated -> requests). Returns the file path. Raises FileNotFoundError (so the
    runner skips) if unavailable and the download fails. The files are large (hundreds of MB); an uploaded copy
    in $ILMARINEN_UPLOADS_DIR is the intended offline path (mirrors qm9/rmd17)."""
    for cand in ("top_tagging.h5", "test.h5", "val.h5", "train.h5"):
        for base in (UPLOADS, CACHE):
            p = os.path.join(base, cand)
            if os.path.exists(p):
                return p
    dst = os.path.join(CACHE, "test.h5")
    try:
        _requests_get_to_file("https://zenodo.org/record/2603256/files/test.h5?download=1", dst)
        _log("top_tagging: downloaded test.h5 from Zenodo")
        return dst
    except Exception as e:
        raise FileNotFoundError(f"Top Quark Tagging: no uploaded/cached HDF5 and Zenodo download failed "
                                f"({str(e)[:60]}). Place test.h5 (or top_tagging.h5) in {UPLOADS}.")
