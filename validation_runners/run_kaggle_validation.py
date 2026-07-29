#!/usr/bin/env python
"""
run_kaggle_validation.py -- fetch an ARBITRARY dataset from Kaggle (via kagglehub), turn it into an
`AllData` pair, route it, train an AllGraph on it, and score it on a held-out split.

Every other runner in this directory trains on a CURATED dataset: run_standard_validation.py iterates the
shared registry (core/dataset_registry + core/extended_datasets), where each entry is a hand-written loader
that already knows its target column, its split, and its SOTA reference. This runner is the opposite -- it
takes a dataset the project has never seen and does the ingestion work itself. Everything downstream of the
`AllData` boundary (contract routing, priced architecture selection, training, evaluation, the skill axis,
the merge-on-write results store) is the SAME code the standard runner uses: this file imports
`add_pipeline_args` / `resolve_pipeline` / `apply_opt_preset` / `make_allgraph` / `_eval_test` / `record_row`
from run_standard_validation rather than re-implementing any of it, so a Kaggle run is configured
identically to a benchmark run.

THREE INGEST MODES (auto-detected; override with --mode):
  tabular  a CSV/TSV/parquet table -> numeric + one-hot feature matrix -> AllData.dense_tensor (rank 2)
  images   a class-directory tree  -> (N, C, hw, hw) float32          -> the SPATIAL contract (rank 4)
  npy      a .npy / .npz array     -> rank passed straight through to the grid-rank router (rank 2-6)

USAGE -- ALWAYS INSPECT FIRST. For an unknown handle you do not know the file inventory, the column names,
or which column is the target, and every guess this script makes is a heuristic:

    python validation_runners/run_kaggle_validation.py --handle uciml/iris --inspect
    python validation_runners/run_kaggle_validation.py --handle uciml/iris --epochs 40

    # tabular regression with a categorical column and missing values
    python validation_runners/run_kaggle_validation.py --handle camnugent/california-housing-prices \\
        --target median_house_value --task regression --target_units USD

    # a competition takes a BARE slug (not "competitions/titanic")
    python validation_runners/run_kaggle_validation.py --source competition --handle titanic \\
        --table train.csv --target Survived --drop_cols Name,Ticket,Cabin

    # an image class-directory tree -> the spatial contract
    python validation_runners/run_kaggle_validation.py --handle tongpython/cat-and-dog \\
        --mode images --hw 64 --per_class 400 --auto_epoch val

    # no Kaggle account / offline: point at an already-extracted directory
    python validation_runners/run_kaggle_validation.py --local_dir /path/to/extracted --inspect

Install:  pip install "ilmarinen[kaggle]"   (kagglehub + pandas + pillow; none is needed by the rest of the
package, and all three are imported lazily, so this file only fails when you actually reach that mode).
Credentials: create an API token at https://www.kaggle.com/settings/api, then either save kaggle.json to
~/.kaggle/kaggle.json or export KAGGLE_USERNAME / KAGGLE_KEY. Check it with --check_auth.

READ THIS BEFORE BELIEVING A NUMBER
  1. A flat tabular feature vector routes to the SEQUENCE contract: rank-2 dense goes to
     discover_mode_structure, which leaves an unstructured vector as a 1D signal, and _fit_sequence then
     reads it as (n, T=n_features, C=1). ilmarinen is a meta-optimizer over computational contracts, not a
     tabular specialist -- EXPECT TO LOSE TO XGBoost/LightGBM ON TABULAR KAGGLE DATA. What this runner is
     for is watching routing and priced selection behave on data nobody tuned them for.
  2. `result["value"]` from fit() is IN-SAMPLE (AllGraph.fit re-scores it on the full training data and says
     so in its own docstring). The number reported here is _eval_test on the held-out split, nothing else.
  3. `chance` is the accuracy of the TRAIN-MAJORITY-CLASS predictor measured on the TEST split -- not 1/K,
     and not the test majority rate. skill = (acc - chance) / (1 - chance). On a 99/1 imbalanced dataset
     accuracy is meaningless; read skill.
  4. LEAKAGE DISCIPLINE: the split is drawn FIRST, and every statistic (median, mean, std, the one-hot
     category set, the label map, per-channel image stats) is fit on the TRAIN rows only. A category seen
     only at test encodes to an all-zero block. This deliberately differs from
     core/extended_datasets.py:load_superconductivity, which z-scores the whole dataset before splitting.
  5. This is NOT a Kaggle leaderboard score. The split is ours (seeded, stratified for classification), not
     the competition's hidden test set -- a competition's own test.csv carries no labels at all.
  6. The target column and the task type are GUESSES. Run --inspect, then pin them with --target / --task.
  7. Encoding is lossy by design: id-like / constant / mostly-missing columns are dropped, categoricals above
     --max_cardinality are dropped, and --max_features drops the widest one-hot blocks. Everything dropped is
     printed and recorded in the results row.
  8. --tensorize_mu (pipeline default 0.05) sends every rank-2 tensor through discover_mode_structure, and
     one-hot blocks are exactly the kind of locally-correlated structure its stride test looks for, so a wide
     table can be PROMOTED to spatial/volumetric -- meaningless for a table. The chosen route is always
     printed and recorded; pass --tensorize_mu -1 to disable tensorization and pin the sequence route.
  9. Kaggle data is unvetted: licences vary per dataset and some require accepting rules in a browser once.
     This script checks neither.
 10. The recurrent sequence primitives are O(T) Python loops, so a table with hundreds of columns trains
     slowly. Cap it with --max_features, or restrict the search with --enabled_contracts.

RESULTS go to a SEPARATE merge-on-write document, `<data-dir>/kaggle_val_rows.json`, never to
standard_val_rows.json -- that file is what `make_results_table.py --insert-readme` renders into the
published benchmark table in README.md, and an arbitrary Kaggle dataset has no business appearing there.
Rows upsert by name, so repeated invocations accumulate. One --handle per invocation (per-dataset flags like
--target / --hw / --x_key do not generalize across handles); batching is deliberately out of scope.

EXIT CODES: 0 ok (or --inspect / --check_auth); 1 the dataset was attempted and failed (an error row is
recorded); 2 a precondition failed (missing package, no credentials, bad handle, undetectable mode, a cap
exceeded) -- nothing is recorded, because nothing was attempted.
"""

import argparse
import gc
import os
import re
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# TWO inserts, deliberately: the repo root so `import ilmarinen` works, and this directory so
# `import run_standard_validation` works under `python -m validation_runners.run_kaggle_validation` as well as
# `python validation_runners/run_kaggle_validation.py`. Under -m, sys.path[0] is the CWD rather than the
# script's own directory, so the single root insert the sibling runners use is not enough by itself.
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The pipeline (flags, preset/device resolution, model construction, evaluation, the results store) is shared
# with the standard runner so a Kaggle run is configured identically to a benchmark run.
from run_standard_validation import (
    BUDGET,
    _eval_test,
    _git_sha,
    _now,
    _seq_depth_for,
    _stub_row,
    add_pipeline_args,
    apply_opt_preset,
    load_results,
    make_allgraph,
    maybe_flatten_grids,
    record_row,
    resolve_pipeline,
)

import ilmarinen
from ilmarinen.core.allgraph import AllData, AllGraph
from ilmarinen.core.paths import cache_path

# --------------------------------------------------------------------------- file-type vocabulary
_TABLE_EXT = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz", ".parquet")
_ARRAY_EXT = (".npy", ".npz")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# A Kaggle competition archive always ships a submission template: the target column is present but the
# feature columns are not. Picking it as the training table is the commonest way this goes wrong, so it is
# excluded from the table candidates outright rather than merely deprioritised.
_SUBMISSION_RE = re.compile(r"(?i)^(sample[_-]?)?submission|^gender_submission")
_TRAIN_RE = re.compile(r"(?i)^train")
_ID_NAME_RE = re.compile(r"(?i)^(unnamed: 0|index|id|idx|row_?id)$|_id$")
_TARGET_NAMES = ("target", "label", "labels", "class", "y", "outcome", "output")
_SPLIT_TRAIN_DIRS = ("train", "training")
_SPLIT_TEST_DIRS = ("test", "testing", "val", "valid", "validation")

# Handle shape per --source. Kaggle uses a different form for each resource type and answers HTTP 403 (not
# 404) for a well-formed handle that does not exist, so checking the shape here turns the commonest user
# error into a precise message instead of an opaque permissions failure.
_HANDLE_RE = {
    "dataset": (re.compile(r"^[\w.-]+/[\w.-]+(/versions/\d+)?$"), "owner/slug  (e.g. uciml/iris)"),
    "competition": (re.compile(r"^[\w.-]+$"), "a BARE slug, no owner (e.g. titanic)"),
    "model": (re.compile(r"^[\w.-]+/[\w.-]+/[\w.-]+/[\w.-]+(/\d+)?$"), "owner/model/framework/variation"),
    "notebook": (re.compile(r"^[\w.-]+/[\w.-]+(/versions/\d+)?$"), "owner/notebook-slug"),
}

_LOCAL_DIR_HINT = " Alternatively, download the data by hand and pass --local_dir <dir>."


def _fail(msg, code=2):
    """Print an actionable precondition failure and return the process exit code. Nothing is recorded: a
    precondition failure means the dataset was never attempted, so it must not appear as a result row."""
    print(f"\nERROR: {msg}\n")
    return code


def _human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- optional dependencies
def _pandas():
    """pandas backs the tabular mode only. Imported lazily (as core/extended_datasets.py does for the Top
    Tagging HDF5) so the other two modes -- and the rest of the package -- never require it."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "tabular mode needs pandas. Install it with:  pip install 'ilmarinen[kaggle]'  (or: pip install "
            "pandas). Use --mode images / --mode npy if this dataset is not a table."
        ) from e
    return pd


def _pillow():
    """Pillow backs the image mode only; same lazy-import rationale as _pandas()."""
    try:
        from PIL import Image, ImageFile
    except ImportError as e:
        raise ImportError(
            "image mode needs Pillow. Install it with:  pip install 'ilmarinen[kaggle]'  (or: pip install pillow)."
        ) from e
    # Do NOT silently half-decode a truncated JPEG. Kaggle image dumps routinely contain corrupt files, and a
    # partially decoded picture is a silently wrong training sample; let it raise so load_images can count it.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    return Image


def _kagglehub():
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is not installed. Install it with:  pip install 'ilmarinen[kaggle]'  (or: pip install "
            "kagglehub). Docs: https://github.com/Kaggle/kagglehub ." + _LOCAL_DIR_HINT
        ) from e
    return kagglehub


# --------------------------------------------------------------------------- acquisition
_AUTH_HELP = (
    "Create an API token at https://www.kaggle.com/settings/api, then EITHER save kaggle.json to "
    "~/.kaggle/kaggle.json (chmod 600), OR export KAGGLE_USERNAME=... KAGGLE_KEY=..., OR export "
    "KAGGLE_API_TOKEN=... . Verify with --check_auth."
)


def _kagglehub_error_text(exc, args):
    """Map a kagglehub failure onto an actionable message. Kaggle answers 403 for BOTH 'no such resource' and
    'you may not have this one', so that branch has to cover the handle form, authentication, and un-accepted
    competition rules at once rather than guessing which of the three it was."""
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401 or "Credential" in name or "Unauthenticated" in name:
        return f"Kaggle rejected the request as unauthenticated ({name}). {_AUTH_HELP}{_LOCAL_DIR_HINT}"
    if status == 403:
        shape = _HANDLE_RE[args.source][1]
        return (
            f"Kaggle returned 403 for handle '{args.handle}' (--source {args.source}). Kaggle uses 403 for "
            f"BOTH 'no such resource' and 'not authorised', so check all three: (1) the handle form for "
            f"--source {args.source} is {shape}; (2) you are authenticated (run --check_auth); (3) for a "
            f"competition or a consent-gated dataset you have accepted its rules once in a browser." + _LOCAL_DIR_HINT
        )
    if status == 404 or "NotFound" in name:
        shape = _HANDLE_RE[args.source][1]
        return (
            f"Kaggle returned 404 for '{args.handle}' -- no such resource. For --source {args.source} the "
            f"handle should look like {shape}." + _LOCAL_DIR_HINT
        )
    if "Corrupt" in name:
        return f"The cached download is corrupt ({name}). Re-run with --force_download." + _LOCAL_DIR_HINT
    return f"Kaggle download failed ({name}): {str(exc)[:200]}" + _LOCAL_DIR_HINT


def _check_auth():
    """Resolve and print the Kaggle identity without downloading anything."""
    try:
        kh = _kagglehub()
    except ImportError as e:
        return _fail(str(e))
    try:
        who = kh.whoami(verbose=False)
    except Exception as e:
        return _fail(f"could not resolve Kaggle credentials ({type(e).__name__}: {str(e)[:120]}). {_AUTH_HELP}")
    print(f"kagglehub {getattr(kh, '__version__', '?')}  authenticated as: {who}")
    return 0


def resolve_source(args):
    """Return the local directory holding the dataset's files, or None on failure (the caller exits 2).

    --local_dir short-circuits kagglehub entirely: no import, no network, no credentials. That is the
    offline / air-gapped path, and it is also what the whole test suite drives.
    """
    if args.local_dir:
        if not os.path.isdir(args.local_dir):
            print(f"\nERROR: --local_dir {args.local_dir!r} is not a directory\n")
            return None
        return os.path.abspath(args.local_dir)
    if not args.handle:
        print("\nERROR: pass --handle <kaggle-handle> (or --local_dir <dir> to skip Kaggle entirely)\n")
        return None
    pat, shape = _HANDLE_RE[args.source]
    if not pat.match(args.handle):
        print(
            f"\nERROR: handle {args.handle!r} does not look like a --source {args.source} handle, which should "
            f"be {shape}. (Competition handles in particular are a bare slug -- 'titanic', not "
            f"'competitions/titanic' and not 'owner/titanic'.)\n"
        )
        return None
    try:
        kh = _kagglehub()
    except ImportError as e:
        print(f"\nERROR: {e}\n")
        return None
    # kagglehub reads KAGGLEHUB_CACHE at call time, so setting it after import is fine. Default the cache
    # under $ILMARINEN_DATA_DIR to keep the project's "no hard-coded scratch paths" contract (core/paths.py);
    # --cache_dir "" opts back out to kagglehub's own ~/.cache/kagglehub.
    if args.cache_dir != "":
        os.environ["KAGGLEHUB_CACHE"] = args.cache_dir or cache_path("kagglehub")
    fn = {
        "dataset": getattr(kh, "dataset_download", None),
        "competition": getattr(kh, "competition_download", None),
        "model": getattr(kh, "model_download", None),
        "notebook": getattr(kh, "notebook_output_download", None),
    }[args.source]
    if fn is None:
        print(f"\nERROR: this kagglehub build has no downloader for --source {args.source}\n")
        return None
    cache = os.environ.get("KAGGLEHUB_CACHE", "kagglehub default")
    print(f"[kaggle] downloading {args.source} '{args.handle}'  (cache: {cache})")
    try:
        kwargs = {"force_download": True} if args.force_download else {}
        path = fn(args.handle, args.file, **kwargs) if args.file else fn(args.handle, **kwargs)
    except Exception as e:
        print(f"\nERROR: {_kagglehub_error_text(e, args)}\n")
        return None
    # With --file, kagglehub returns the path TO THAT FILE rather than the version directory.
    root = path if os.path.isdir(path) else os.path.dirname(path)
    print(f"[kaggle] -> {root}")
    return root


# --------------------------------------------------------------------------- inventory / mode detection
def walk_inventory(root):
    """[(relpath, size_bytes)] for every real file under `root`, largest first. Skips macOS archive cruft and
    dotfiles, which otherwise show up as phantom 'tables' in an unpacked Kaggle zip."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            try:
                size = os.stat(full).st_size
            except OSError:
                continue
            out.append((os.path.relpath(full, root), size))
    out.sort(key=lambda t: -t[1])
    return out


def _has_ext(rel, exts):
    low = rel.lower()
    return any(low.endswith(e) for e in exts)


def classify_files(inv):
    """Bucket the inventory by extension. Submission templates are excluded from `tables` here rather than
    later, so they can neither win the largest-table tie-break nor trip the tabular branch of detect_mode."""
    tables, arrays, images, other = [], [], [], []
    for rel, size in inv:
        base = os.path.basename(rel)
        if _has_ext(rel, _TABLE_EXT):
            (other if _SUBMISSION_RE.match(base) else tables).append((rel, size))
        elif _has_ext(rel, _ARRAY_EXT):
            arrays.append((rel, size))
        elif _has_ext(rel, _IMAGE_EXT):
            images.append((rel, size))
        else:
            other.append((rel, size))
    return {"tables": tables, "arrays": arrays, "images": images, "other": other}


def _image_dir_counts(images):
    """{directory relpath: number of images directly inside it}."""
    counts = {}
    for rel, _ in images:
        d = os.path.dirname(rel)
        counts[d] = counts.get(d, 0) + 1
    return counts


def _class_dir_roots(counts):
    """{parent relpath: {child dirname: n_images}} for every directory whose IMMEDIATE subdirectories hold the
    images -- i.e. every candidate `<root>/<class>/*.png` layout."""
    parents = {}
    for d, n in counts.items():
        parent, child = os.path.dirname(d), os.path.basename(d)
        if child:
            parents.setdefault(parent, {})[child] = n
    return parents


def detect_mode(root, buckets, args):
    """(mode, reason), where mode is None when nothing matched. --mode always wins; otherwise the first
    matching rule below, in this fixed order.

    Images are tested BEFORE tables because image datasets routinely ship a metadata CSV alongside the
    pictures. The >=2-class-directories AND >=--min_images guard is what stops a tabular dataset that happens
    to contain a handful of plot PNGs from being misrouted into the image path.
    """
    if args.mode != "auto":
        return args.mode, f"--mode {args.mode}"
    counts = _image_dir_counts(buckets["images"])
    n_img = sum(counts.values())
    parents = _class_dir_roots(counts)
    best_parent, best_children = max(parents.items(), key=lambda kv: sum(kv[1].values()), default=(None, {}))
    if best_parent is not None and len(best_children) >= 2 and n_img >= args.min_images:
        return "images", f"{len(best_children)} class directories under {best_parent or '<root>'} ({n_img} images)"
    if len(counts) == 1 and n_img >= args.min_images and (buckets["tables"] or args.labels_csv):
        return "images", f"{n_img} images in one directory plus a labels table"
    if buckets["arrays"] and not buckets["tables"]:
        return "npy", f"{len(buckets['arrays'])} array file(s), no table"
    if buckets["tables"]:
        return "tabular", f"{len(buckets['tables'])} table candidate(s)"
    if buckets["arrays"]:
        return "npy", f"{len(buckets['arrays'])} array file(s) (every table was a submission template)"
    return None, "no table, no array, and no image-directory layout"


# --------------------------------------------------------------------------- tabular
def pick_table(root, buckets, args):
    """(abs path, relpath) of the training table. Explicit --table wins (matched on relpath or basename);
    otherwise prefer a `train*` basename, then the largest remaining candidate."""
    tables = buckets["tables"]
    if args.table:
        want = args.table.replace("\\", "/")
        for rel, _ in tables + buckets["other"]:
            if rel.replace("\\", "/") == want or os.path.basename(rel) == want:
                return os.path.join(root, rel), rel
        raise FileNotFoundError(f"--table {args.table!r} not found. Tables present: {[r for r, _ in tables] or 'none'}")
    if not tables:
        raise FileNotFoundError("no table found (submission templates are excluded); pass --table or --mode")
    trains = [t for t in tables if _TRAIN_RE.match(os.path.basename(t[0]))]
    rel = (trains or tables)[0][0]
    return os.path.join(root, rel), rel


def read_table(path, args, nrows=None):
    """Read a table into a DataFrame. sep=None plus the python engine sniffs , ; tab and |, so a European CSV
    works without a flag; utf-8 falls back to latin-1, which is what most non-UTF8 Kaggle exports are."""
    pd = _pandas()
    if path.lower().endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(f"reading .parquet needs pyarrow (pip install pyarrow): {e}") from e
        return df.head(nrows) if nrows else df
    kw = {"low_memory": False}
    if args.sep:
        kw["sep"] = args.sep
    else:
        kw["sep"], kw["engine"] = None, "python"
        kw.pop("low_memory")  # the python engine does not accept low_memory
    if nrows:
        kw["nrows"] = int(nrows)
    try:
        return pd.read_csv(path, encoding=args.encoding or "utf-8", **kw)
    except UnicodeDecodeError:
        if args.encoding:
            raise
        print("[table] utf-8 decode failed; retrying as latin-1")
        return pd.read_csv(path, encoding="latin-1", **kw)


def _is_numeric_col(s):
    """True when the column is usable as a number. A column of digit strings with a handful of blanks (the
    Telco `TotalCharges` case) counts: coercion recovers >=90% of the non-null cells."""
    pd = _pandas()
    if s.dtype.kind in "biufc":
        return True
    n_valid = int(s.notna().sum())
    if n_valid == 0:
        return False
    return int(pd.to_numeric(s, errors="coerce").notna().sum()) >= 0.9 * n_valid


def _col_drop_reason(s, n_rows, name, args):
    """The single predicate both the encoder and the --inspect profile use, so the preview can never drift
    from what the real run does. Returns a reason string, or None to keep the column."""
    if str(name) in args.drop_set:
        return "--drop_cols"
    card = int(s.nunique(dropna=True))
    if card <= 1:
        return "constant"
    miss = float(s.isna().mean())
    if miss > args.max_missing:
        return f"{miss:.0%} missing (> --max_missing {args.max_missing:.0%})"
    # An identifier carries no signal and, when it tracks row order, leaks the split. Float columns are
    # exempt from the all-unique test: a continuous measurement is legitimately unique per row.
    if _ID_NAME_RE.match(str(name)) or (card == n_rows and s.dtype.kind != "f"):
        return "id-like (unique per row)"
    if not _is_numeric_col(s) and card > args.max_cardinality:
        return f"categorical, cardinality {card} > --max_cardinality {args.max_cardinality}"
    return None


def guess_target(df, args):
    """(column, why). --target wins; then a conventionally-named column; then the last column, which is the
    near-universal CSV convention."""
    cols = list(df.columns)
    if args.target:
        if args.target not in cols:
            raise KeyError(f"--target {args.target!r} is not a column of this table. Columns: {cols}")
        return args.target, "--target"
    for c in cols:
        if str(c).strip().lower() in _TARGET_NAMES:
            return c, f"column {c!r} matches the conventional target names"
    return cols[-1], "last column (no column matched the conventional target names)"


def infer_task(s, args):
    """(task, why) for a pandas target column. --task wins. A non-numeric target is a classification; a
    numeric one is a classification only when it looks like a code: integral, and few distinct values both in
    absolute terms and relative to n."""
    if args.task != "auto":
        return args.task, "--task"
    card = int(s.nunique(dropna=True))
    if card < 2:
        raise ValueError(f"target column {s.name!r} has {card} distinct value(s) -- there is nothing to predict")
    if not _is_numeric_col(s):
        if card > args.max_classes:
            raise ValueError(
                f"target {s.name!r} is non-numeric with {card} distinct values (> --max_classes "
                f"{args.max_classes}). Pick a different --target, or raise --max_classes."
            )
        return "classification", f"non-numeric dtype, {card} distinct values"
    v = _pandas().to_numeric(s, errors="coerce").dropna().to_numpy()
    if _integral(v) and card <= args.max_classes and card <= max(20, 0.05 * len(s)):
        return "classification", f"integral numeric, {card} distinct values <= --max_classes {args.max_classes}"
    return "regression", f"numeric with {card} distinct values"


def infer_task_array(y, args):
    """infer_task's rules for a bare numpy target, so the array mode never pulls in pandas."""
    if args.task != "auto":
        return args.task, "--task"
    card = int(len(np.unique(y)))
    if card < 2:
        raise ValueError(f"target has {card} distinct value(s) -- there is nothing to predict")
    if y.dtype.kind in "USOb":
        if card > args.max_classes:
            raise ValueError(f"non-numeric target with {card} distinct values > --max_classes {args.max_classes}")
        return "classification", f"non-numeric dtype, {card} distinct values"
    if _integral(y) and card <= args.max_classes and card <= max(20, 0.05 * len(y)):
        return "classification", f"integral, {card} distinct values <= --max_classes {args.max_classes}"
    return "regression", f"numeric with {card} distinct values"


def _integral(v):
    v = np.asarray(v, dtype=np.float64)
    return bool(v.size and np.all(np.isfinite(v)) and np.all(np.equal(np.mod(v, 1), 0)))


def profile_table(df, target, args):
    """Per-column report for --inspect. Verdicts come from the same predicates the encoder uses, so the
    preview cannot disagree with the run."""
    rows, n = [], len(df)
    for c in df.columns:
        s = df[c]
        card = int(s.nunique(dropna=True))
        if c == target:
            verdict = "TARGET"
        else:
            why = _col_drop_reason(s, n, c, args)
            verdict = f"DROP {why}" if why else ("numeric" if _is_numeric_col(s) else f"one-hot (card {card})")
        rows.append(
            {
                "name": str(c),
                "dtype": str(s.dtype),
                "cardinality": card,
                "missing": int(s.isna().sum()),
                "samples": ", ".join(str(v)[:18] for v in s.dropna().unique()[:3]),
                "verdict": verdict,
            }
        )
    return rows


def _drop_useless_columns(feat, args):
    """(kept DataFrame, [(column, why)]). Applied before the split -- these predicates read only shape and
    cardinality, never a value distribution, so they carry no test information into training."""
    n = len(feat)
    dropped = [(str(c), _col_drop_reason(feat[c], n, c, args)) for c in feat.columns]
    dropped = [(c, why) for c, why in dropped if why]
    names = {c for c, _ in dropped}
    return feat[[c for c in feat.columns if str(c) not in names]], dropped


def _as_strings(s):
    """Column -> numpy array of str, with missing values collapsed onto one explicit level. Giving NaN its own
    level (rather than dropping the row) keeps 'the value was absent' as usable signal, which on Kaggle data
    it very often is."""
    return np.where(s.isna().to_numpy(), "__nan__", s.astype(str).to_numpy())


def _fit_encoder(tr, args):
    """Fit the feature encoder on the TRAIN rows ONLY: per numeric column a median (for imputation) and a
    mean/std (for the z-score), and per categorical column the sorted set of levels. Nothing here may see a
    test row -- that is the entire point, and test_scaler_is_fit_on_train_only pins it.

    Returns {"numeric": [(col, median, mean, std)], "cats": [(col, [level, ...])], "names": [...],
             "dropped": [(col, why)]}.
    """
    pd = _pandas()
    numeric, cats, dropped = [], [], []
    for c in tr.columns:
        s = tr[c]
        if _is_numeric_col(s):
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                dropped.append((str(c), "no finite values in the train split"))
                continue
            numeric.append((c, float(np.median(finite)), float(finite.mean()), float(max(finite.std(), 1e-6))))
        else:
            levels = sorted(set(_as_strings(s).tolist()))
            if len(levels) > args.max_cardinality:
                dropped.append((str(c), f"cardinality {len(levels)} > --max_cardinality on the train split"))
                continue
            cats.append((c, levels))
    # Width cap. The sequence contract reads the feature axis as a time axis and its recurrent primitives are
    # O(T) Python loops, so an unbounded one-hot expansion is a training-TIME cliff, not just a memory one.
    # Drop the widest blocks first: they buy the most axis per column of original information.
    width = len(numeric) + sum(len(lv) for _, lv in cats)
    while width > args.max_features and cats:
        cats.sort(key=lambda t: -len(t[1]))
        c, lv = cats.pop(0)
        dropped.append((str(c), f"one-hot width {len(lv)} dropped to fit --max_features {args.max_features}"))
        width -= len(lv)
    if width > args.max_features:
        raise ValueError(
            f"{width} numeric features exceed --max_features {args.max_features}, with no categorical column "
            f"left to drop. Raise --max_features, or narrow the table with --drop_cols."
        )
    names = [str(c) for c, *_ in numeric] + [f"{c}={lv}" for c, levels in cats for lv in levels]
    return {"numeric": numeric, "cats": cats, "names": names, "dropped": dropped}


def _apply_encoder(df, enc, standardize):
    """Encode with an ALREADY-FITTED encoder. A category absent from the train split encodes to an all-zero
    block -- the standard unknown-level encoding. Nothing here re-fits, so it is safe on the test split."""
    pd = _pandas()
    n, blocks = len(df), []
    for c, med, mean, std in enc["numeric"]:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        v = np.where(np.isfinite(v), v, med)
        blocks.append(((v - mean) / std if standardize else v)[:, None])
    for c, levels in enc["cats"]:
        idx = {lv: i for i, lv in enumerate(levels)}
        block = np.zeros((n, len(levels)), np.float64)
        for r, val in enumerate(_as_strings(df[c])):
            j = idx.get(val)
            if j is not None:
                block[r, j] = 1.0
        blocks.append(block)
    if not blocks:
        raise ValueError("every feature column was dropped -- nothing left to train on (see the dropped list)")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def random_split(n, test_frac, seed):
    if n < 2:
        raise ValueError(f"need at least 2 samples to split; got {n}")
    idx = np.random.RandomState(seed).permutation(n)
    k = min(max(1, int(round((1.0 - test_frac) * n))), n - 1)
    return idx[:k], idx[k:]


def stratified_split(y, test_frac, seed):
    """Per-class seeded split. An arbitrary Kaggle target is routinely imbalanced, and an unstratified draw
    can leave a minority class absent from train (so it can never be predicted) or absent from test (so its
    error is invisible). Every class with >=2 rows keeps at least one row on each side."""
    rs = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        ids = ids[rs.permutation(len(ids))]
        n_te = min(max(int(round(test_frac * len(ids))), 1), len(ids) - 1) if len(ids) > 1 else 0
        te.extend(ids[:n_te].tolist())
        tr.extend(ids[n_te:].tolist())
    return np.array(sorted(tr), dtype=int), np.array(sorted(te), dtype=int)


def _encode_labels(values):
    """Raw label values -> (contiguous int64 codes, level names), dropping any class with fewer than 2 rows so
    the stratified split can always place one on each side. Codes are recomputed after the drop, so they stay
    contiguous from zero -- which _infer_nout and CrossEntropyLoss both require."""
    levels, y = np.unique(values, return_inverse=True)
    counts = np.bincount(y, minlength=len(levels))
    dropped = [str(levels[i]) for i in np.flatnonzero(counts < 2)]
    keep = counts[y] >= 2
    if dropped:
        levels, y = np.unique(levels[y[keep]], return_inverse=True)
    if len(levels) < 2:
        raise ValueError(f"fewer than 2 usable classes remain (dropped singletons: {dropped})")
    return y.astype(np.int64), [str(v) for v in levels], keep, dropped


def _finish(Xtr, ytr, Xte, yte, task, args, meta, target_scale=None, target_units=None):
    """Wrap encoded arrays into the registry-shaped dict every runner consumes (the loader contract is
    documented at the top of core/dataset_registry). `chance` is the trivial predictor's score MEASURED ON
    TEST: for a classification that is the TRAIN-majority class scored against the test labels -- not 1/K and
    not the test majority rate, both of which flatter the model on an imbalanced dataset -- and for a
    regression it is 0.0, the R2 of the mean predictor."""
    if task == "classification":
        maj = int(np.bincount(ytr.astype(int), minlength=int(meta["n_classes"])).argmax())
        chance = float((yte.astype(int) == maj).mean())
        meta["majority_class"] = meta["class_names"][maj] if meta.get("class_names") else maj
    else:
        chance = 0.0
    meta["n_train"], meta["n_test"] = int(len(ytr)), int(len(yte))
    meta["n_features"] = int(np.prod(Xtr.shape[1:]))
    d = {
        "train": AllData.dense_tensor(torch.tensor(Xtr), y=ytr, kind_hint=args.kind_hint),
        "test": AllData.dense_tensor(torch.tensor(Xte), y=yte, kind_hint=args.kind_hint),
        "task": task,
        "chance": chance,
        "field": args.field or f"kaggle ({meta.get('handle') or meta.get('root')})",
        "sota": args.sota or "n/a -- user-supplied dataset, no reference number",
        "meta": meta,
    }
    if target_scale is not None:
        d["target_scale"] = float(target_scale)
        d["target_units"] = target_units or "target"
    return d


def encode_tabular(root, buckets, args):
    """A CSV/TSV/parquet table -> a (n, n_features) dense AllData pair."""
    pd = _pandas()
    path, rel = pick_table(root, buckets, args)
    df = read_table(path, args, nrows=args.max_rows)
    print(f"[table] {rel}  rows={len(df)} cols={len(df.columns)}")
    if len(df) < 4:
        raise ValueError(f"table {rel!r} has only {len(df)} rows -- too few to split and train")
    target, t_why = guess_target(df, args)
    task, task_why = infer_task(df[target], args)
    print(f"[table] target={target!r} ({t_why});  task={task} ({task_why})")

    n0 = len(df)
    df = df[df[target].notna()]
    if len(df) < n0:
        print(f"[table] dropped {n0 - len(df)} row(s) with a missing target")
    feat, dropped = _drop_useless_columns(df.drop(columns=[target]), args)
    for c, why in dropped:
        print(f"[table] drop column {c!r}: {why}")
    if feat.shape[1] == 0:
        raise ValueError("every feature column was dropped; relax --max_cardinality / --max_missing / --drop_cols")

    meta = {"table": rel, "target": str(target), "target_why": t_why, "task_why": task_why}
    if task == "classification":
        y, levels, keep, singletons = _encode_labels(_as_strings(df[target]))
        if singletons:
            print(f"[table] dropped {len(singletons)} class(es) with <2 rows: {singletons}")
            feat = feat[keep]
        meta["n_classes"], meta["class_names"] = len(levels), levels
        tr, te = stratified_split(y, args.test_frac, args.split_seed)
        ytr, yte, scale, units = y[tr], y[te], None, None
    else:
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(y)
        if not finite.all():
            print(f"[table] dropped {int((~finite).sum())} row(s) with a non-numeric target")
            feat, y = feat[finite], y[finite]
        tr, te = random_split(len(y), args.test_frac, args.split_seed)
        # Standardize with TRAIN statistics and keep the train std as target_scale, so _eval_test reports the
        # MAE back in the original units. The mean cancels inside |pred - y|, so that inversion is exact.
        mu, scale = float(y[tr].mean()), float(max(y[tr].std(), 1e-12))
        ytr = ((y[tr] - mu) / scale).astype(np.float32)
        yte = ((y[te] - mu) / scale).astype(np.float32)
        units = args.target_units or str(target)
        meta["n_classes"], meta["target_mean"], meta["target_std"] = None, mu, scale

    enc = _fit_encoder(feat.iloc[tr], args)
    for c, why in enc["dropped"]:
        print(f"[table] drop column {c!r}: {why}")
    Xtr = _apply_encoder(feat.iloc[tr], enc, not args.no_standardize)
    Xte = _apply_encoder(feat.iloc[te], enc, not args.no_standardize)
    meta["dropped_columns"] = [f"{c} ({why})" for c, why in dropped + enc["dropped"]]
    meta["numeric_columns"] = [str(c) for c, *_ in enc["numeric"]]
    meta["onehot_columns"] = [f"{c}({len(lv)})" for c, lv in enc["cats"]]
    meta["feature_names"] = enc["names"]
    print(f"[table] encoded {len(enc['numeric'])} numeric + {len(enc['cats'])} one-hot -> {Xtr.shape[1]} features")
    return _finish(Xtr, ytr, Xte, yte, task, args, meta, scale, units)


# --------------------------------------------------------------------------- images
def detect_image_layout(root, buckets, args):
    """(layout, payload). One of:
    class_dirs        <dir>/<class>/*.png
    split_class_dirs  <dir>/train/<class>/* plus a test|val sibling -- the dataset's OWN split
    flat_csv          one image directory plus a labels table
    """
    counts = _image_dir_counts(buckets["images"])
    if not counts:
        raise FileNotFoundError("no image files found under the dataset root")
    parents = _class_dir_roots(counts)
    if args.image_root is not None:
        rel = args.image_root.strip("/")
        if rel not in parents:
            raise FileNotFoundError(
                f"--image_root {args.image_root!r} has no image subdirectories. Directories holding images: "
                f"{sorted(counts)}"
            )
        return "class_dirs", {"dir": rel}
    # Honour the dataset's own train/test split when it shipped one: it is strictly better than re-splitting,
    # because the author may have split by subject or session rather than at random. A shipped split shows up
    # as two SIBLING class-directory roots (keys of `parents`, not children of one) named train/ and test|val/
    # under a common parent -- e.g. root/train/<class>/*.png beside root/test/<class>/*.png.
    if args.use_dir_split != "off":
        groups = {}
        for p in parents:
            groups.setdefault(os.path.dirname(p), {})[os.path.basename(p).lower()] = p
        for _, sibs in sorted(groups.items()):
            tr = next((sibs[k] for k in _SPLIT_TRAIN_DIRS if k in sibs), None)
            te = next((sibs[k] for k in _SPLIT_TEST_DIRS if k in sibs), None)
            if tr is not None and te is not None and len(parents.get(tr, {})) >= 2 and len(parents.get(te, {})) >= 2:
                return "split_class_dirs", {"train": tr, "test": te}
    best_parent, best_children = max(parents.items(), key=lambda kv: sum(kv[1].values()), default=(None, {}))
    if best_parent is not None and len(best_children) >= 2:
        return "class_dirs", {"dir": best_parent}
    if buckets["tables"] or args.labels_csv:
        img_dir = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
        csv_rel = args.labels_csv or pick_table(root, buckets, args)[1]
        return "flat_csv", {"dir": img_dir, "csv": csv_rel}
    raise FileNotFoundError(
        f"could not read an image layout: directories holding images are {sorted(counts)}, and none of them "
        f"has >=2 class subdirectories. Pass --image_root <dir>, or --labels_csv with --image_col/--label_col."
    )


def list_class_images(root, rel_dir, args):
    """{class_name: [abs path, ...]} with a deterministic, seeded per-class subsample. Sorting first and then
    permuting with a fixed seed makes the cap reproducible AND unbiased -- taking the alphabetical head would
    silently select by filename, which on many datasets encodes the capture session."""
    base = os.path.join(root, rel_dir) if rel_dir else root
    rs = np.random.RandomState(args.split_seed)
    out = {}
    for cls in sorted(os.listdir(base)):
        d = os.path.join(base, cls)
        if cls.startswith(".") or not os.path.isdir(d):
            continue
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if _has_ext(f, _IMAGE_EXT))
        if not paths:
            continue
        if args.per_class and len(paths) > args.per_class:
            paths = [paths[i] for i in sorted(rs.permutation(len(paths))[: args.per_class])]
        out[cls] = paths
    if len(out) < 2:
        raise ValueError(f"need >=2 class directories under {rel_dir or '<root>'}; found {sorted(out)}")
    return out


def load_images(paths, args):
    """(array (n_ok, C, hw, hw) float32 in [0,1], indices of the paths that decoded). Returning the kept
    indices rather than a count is what lets the caller drop the FAILED sample's own label instead of shifting
    every later label by one -- Kaggle image dumps do contain truncated files, and a silent off-by-one in the
    label alignment would be invisible in the reported accuracy."""
    Image = _pillow()
    n_c = 1 if args.gray else 3
    buf, kept = np.zeros((len(paths), n_c, args.hw, args.hw), np.float32), []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("L" if args.gray else "RGB").resize((args.hw, args.hw), Image.BILINEAR)
                a = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            continue
        buf[len(kept)] = a[None] if args.gray else a.transpose(2, 0, 1)
        kept.append(i)
    return buf[: len(kept)], np.asarray(kept, dtype=int)


def _image_memory_guard(n, args):
    mb = n * (1 if args.gray else 3) * args.hw * args.hw * 4 / 1e6
    if mb > args.max_mb:
        raise ValueError(
            f"{n} images at {args.hw}x{args.hw}x{1 if args.gray else 3} float32 need ~{mb:.0f} MB > --max_mb "
            f"{args.max_mb:.0f}. Lower --per_class or --hw (cost scales as per_class * hw^2), or raise --max_mb."
        )


def _load_split(root, rel_dir, classes, args):
    """One class-directory tree -> (X, y) with labels aligned to `classes`."""
    split = list_class_images(root, rel_dir, args)
    paths, labels = [], []
    for cls in classes:
        ps = split.get(cls, [])
        paths.extend(ps)
        labels.extend([classes.index(cls)] * len(ps))
    _image_memory_guard(len(paths), args)
    X, kept = load_images(paths, args)
    return X, np.asarray(labels, np.int64)[kept], len(paths) - len(kept)


def encode_images(root, buckets, args):
    """A class-directory tree -> (N, C, hw, hw) float32, which route_grid_rank reads as rank 4 -> SPATIAL.

    --hw is a single int on purpose: build_spatial_schema takes ONE hw (models/spatial_schema.py), so a
    non-square resize would build a network sized for the wrong axis.
    """
    layout, payload = detect_image_layout(root, buckets, args)
    print(f"[images] layout={layout} {payload}")
    meta = {"image_layout": layout, "hw": int(args.hw), "gray": bool(args.gray), "per_class": args.per_class}
    meta.update({f"image_{k}": v for k, v in payload.items()})

    if layout == "flat_csv":
        Xtr, ytr, Xte, yte, classes, failed = _encode_flat_csv(root, payload, args)
        meta["split"] = "stratified (a flat directory ships no split)"
    elif layout == "class_dirs":
        classes = sorted(list_class_images(root, payload["dir"], args))
        X, y, failed = _load_split(root, payload["dir"], classes, args)
        tr, te = stratified_split(y, args.test_frac, args.split_seed)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        meta["split"] = "stratified (the dataset shipped no split)"
    else:
        classes = sorted(
            set(list_class_images(root, payload["train"], args)) | set(list_class_images(root, payload["test"], args))
        )
        Xtr, ytr, f1 = _load_split(root, payload["train"], classes, args)
        Xte, yte, f2 = _load_split(root, payload["test"], classes, args)
        failed = f1 + f2
        meta["split"] = f"the dataset's own {payload['train']} / {payload['test']} directories"

    if failed:
        print(f"[images] skipped {failed} unreadable/corrupt image(s)")
    if len(Xtr) < 2 or len(Xte) < 1:
        raise ValueError(f"too few usable images after decoding ({len(Xtr)} train / {len(Xte)} test)")
    if not args.no_standardize:
        mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)  # train-only channel statistics
        sd = np.maximum(Xtr.std(axis=(0, 2, 3), keepdims=True), 1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    meta["n_classes"], meta["class_names"], meta["images_skipped"] = len(classes), list(classes), int(failed)
    print(f"[images] {len(Xtr)} train / {len(Xte)} test  shape={tuple(Xtr.shape[1:])}  classes={len(classes)}")
    return _finish(Xtr, ytr, Xte, yte, "classification", args, meta)


def _encode_flat_csv(root, payload, args):
    """One image directory plus a labels table: the CSV column whose values match the image basenames is the
    filename column, and the first low-cardinality column after it is the label."""
    df = read_table(os.path.join(root, payload["csv"]), args, nrows=args.max_rows)
    img_dir = os.path.join(root, payload["dir"])
    present = {f for f in os.listdir(img_dir) if _has_ext(f, _IMAGE_EXT)}
    stems = {os.path.splitext(f)[0]: f for f in present}

    def match_rate(col):
        vals = df[col].astype(str)
        return float(np.mean([(v in present) or (os.path.splitext(v)[0] in stems) for v in vals])) if len(vals) else 0.0

    img_col = args.image_col
    if img_col is None:
        rates = sorted(((match_rate(c), str(c)) for c in df.columns), reverse=True)
        if not rates or rates[0][0] < 0.8:
            best = f"{rates[0][0]:.0%} for {rates[0][1]!r}" if rates else "none"
            raise ValueError(
                f"no column of {payload['csv']} matches the image filenames (best match: {best}). Pass "
                f"--image_col and --label_col."
            )
        img_col = rates[0][1]
    lab_col = args.label_col
    if lab_col is None:
        cands = [c for c in df.columns if str(c) != str(img_col) and 2 <= int(df[c].nunique()) <= args.max_classes]
        if not cands:
            raise ValueError(f"no usable label column found in {payload['csv']}; pass --label_col")
        lab_col = cands[0]
    print(f"[images] labels from {payload['csv']}: image_col={img_col!r} label_col={lab_col!r}")

    paths, labs = [], []
    for v, lab in zip(df[img_col].astype(str), _as_strings(df[lab_col])):
        fn = v if v in present else stems.get(os.path.splitext(v)[0])
        if fn:
            paths.append(os.path.join(img_dir, fn))
            labs.append(lab)
    if len(paths) < 4:
        raise ValueError(f"only {len(paths)} row(s) of {payload['csv']} resolved to an existing image file")
    y, classes, keep, singletons = _encode_labels(np.asarray(labs))
    paths = [p for p, k in zip(paths, keep) if k]
    if singletons:
        print(f"[images] dropped {len(singletons)} class(es) with <2 rows: {singletons}")
    if args.per_class:
        rs = np.random.RandomState(args.split_seed)
        sel = np.concatenate(
            [np.flatnonzero(y == c)[rs.permutation(int((y == c).sum()))][: args.per_class] for c in range(len(classes))]
        )
        sel.sort()
        paths, y = [paths[i] for i in sel], y[sel]
    _image_memory_guard(len(paths), args)
    X, kept = load_images(paths, args)
    y = y[kept]
    tr, te = stratified_split(y, args.test_frac, args.split_seed)
    return X[tr], y[tr], X[te], y[te], classes, len(paths) - len(kept)


# --------------------------------------------------------------------------- npy / npz
_X_KEYS = ("x", "X", "data", "images", "features", "inputs", "arr_0")
_Y_KEYS = ("y", "Y", "label", "labels", "target", "targets", "classes", "arr_1")
_Y_FILES = ("y.npy", "labels.npy", "targets.npy")


def pick_array(root, buckets, args):
    if args.array:
        want = args.array.replace("\\", "/")
        for rel, _ in buckets["arrays"]:
            if rel.replace("\\", "/") == want or os.path.basename(rel) == want:
                return os.path.join(root, rel), rel
        raise FileNotFoundError(f"--array {args.array!r} not found. Arrays: {[r for r, _ in buckets['arrays']]}")
    if not buckets["arrays"]:
        raise FileNotFoundError("no .npy/.npz file found")
    rel = buckets["arrays"][0][0]
    return os.path.join(root, rel), rel


def load_arrays(path, args):
    """(X, y, provenance) from a .npy/.npz. Keys are matched by convention, then by shape; --x_key / --y_key /
    --y_file always win."""
    if path.lower().endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        xk = args.x_key or next((k for k in _X_KEYS if k in keys), None)
        if xk is None:
            xk = max(keys, key=lambda k: (z[k].ndim, z[k].size))
        if xk not in keys:
            raise KeyError(f"--x_key {xk!r} is not in {os.path.basename(path)} (keys: {keys})")
        yk = args.y_key or next((k for k in _Y_KEYS if k in keys and k != xk), None)
        if yk is None and len(keys) == 2:
            other = next(k for k in keys if k != xk)
            if z[other].ndim <= 2 and len(z[other]) == len(z[xk]):
                yk = other
        if yk is None or yk not in keys:
            raise ValueError(
                f"no target array found in {os.path.basename(path)} (keys: {keys}). Pass --y_key. Unsupervised "
                f"data is out of scope for this runner."
            )
        return np.asarray(z[xk]), np.asarray(z[yk]), {"x_key": xk, "y_key": yk}
    X = np.load(path, allow_pickle=False)
    stem = os.path.splitext(os.path.basename(path))[0]
    cands = (
        [args.y_file]
        if args.y_file
        else [os.path.join(os.path.dirname(path), f) for f in (*_Y_FILES, f"{stem}_y.npy", f"{stem}_labels.npy")]
    )
    for c in cands:
        if c and os.path.exists(c):
            return np.asarray(X), np.asarray(np.load(c, allow_pickle=False)), {"y_file": os.path.basename(c)}
    raise ValueError(
        f"{os.path.basename(path)} holds features but no labels were found beside it (looked for {_Y_FILES}). "
        f"Pass --y_file <path>, or use a .npz carrying both arrays."
    )


def encode_arrays(root, buckets, args):
    """A raw array -> AllData with its rank passed straight through to the grid-rank router."""
    path, rel = pick_array(root, buckets, args)
    X, y, keys = load_arrays(path, args)
    print(f"[array] {rel}  X{X.shape} {X.dtype}  y{y.shape} {y.dtype}  {keys}")
    if X.ndim < 2 or X.ndim > 6:
        hint = "reshape it to (n_samples, n_features)" if X.ndim < 2 else "collapse axes, or pass --kind_hint"
        raise ValueError(
            f"X has rank {X.ndim}; route_grid_rank only handles rank 2-6 (2=flat, 3=sequence, 4=spatial, "
            f"5=volumetric, 6=4d). {hint}."
        )
    if len(y) != len(X):
        raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
    if y.ndim == 2 and y.shape[1] > 1 and np.allclose(y.sum(1), 1.0):
        print(f"[array] y looks one-hot ({y.shape[1]} columns) -> argmax to integer labels")
        y = y.argmax(1)
    y = np.asarray(y).squeeze()
    task, why = infer_task_array(y, args)
    print(f"[array] task={task}  ({why})")

    meta = {"array": rel, "task_why": why, **{f"array_{k}": v for k, v in keys.items()}}
    X = X.astype(np.float32, copy=False)
    if task == "classification":
        codes, classes, keep, singletons = _encode_labels(y)
        if singletons:
            print(f"[array] dropped {len(singletons)} class(es) with <2 rows: {singletons}")
            X = X[keep]
        meta["n_classes"], meta["class_names"] = len(classes), classes
        tr, te = stratified_split(codes, args.test_frac, args.split_seed)
        ytr, yte, scale, units = codes[tr], codes[te], None, None
    else:
        yf = np.asarray(y, dtype=np.float64)
        tr, te = random_split(len(yf), args.test_frac, args.split_seed)
        mu, scale = float(yf[tr].mean()), float(max(yf[tr].std(), 1e-12))
        ytr, yte = ((yf[tr] - mu) / scale).astype(np.float32), ((yf[te] - mu) / scale).astype(np.float32)
        units = args.target_units or "target"
        meta["n_classes"], meta["target_mean"], meta["target_std"] = None, mu, scale
    Xtr, Xte = _normalize_arrays(X[tr], X[te], args, meta)
    return _finish(Xtr, ytr, Xte, yte, task, args, meta, scale, units)


def _normalize_arrays(Xtr, Xte, args, meta):
    """Train-only feature normalization. `auto` z-scores a flat matrix (the tabular-like case) and rescales a
    byte-valued grid into [0,1]; anything else is assumed already to be in a sensible range."""
    how = args.normalize
    if how == "auto":
        how = "zscore" if Xtr.ndim == 2 else ("unit" if float(Xtr.max()) > 1.5 else "none")
    meta["normalize"] = how
    if how == "zscore":
        mu, sd = Xtr.mean(0, keepdims=True), np.maximum(Xtr.std(0, keepdims=True), 1e-6)
        return (Xtr - mu) / sd, (Xte - mu) / sd
    if how == "unit":
        hi = float(max(Xtr.max(), 1e-6))
        return Xtr / hi, Xte / hi
    return Xtr, Xte


# --------------------------------------------------------------------------- pipeline
def probe_contract(train, args, tzmu, router, enabled_sg):
    """Route WITHOUT training, so the per-contract BUDGET key is known before make_allgraph is called.

    This is faithful for this runner because all three ingest paths build a dense AllData with no node_feats
    and no positions, and every routing layer that could disagree with route() -- tiebreak, symmetry_routing,
    equivariance discovery -- is positions-gated and short-circuits, so _resolve_contract falls straight
    through to route() at fit time. The two knobs that DO change the answer, tensorize_mu (the tensorization
    price) and equivariant_if_positions, are set here exactly as make_allgraph sets them. The one genuine
    gap: route() does not apply the --enabled_contracts restriction that fit() applies, so that is replayed
    explicitly below.
    """
    if args.budget_contract:
        return args.budget_contract, {"why": "--budget_contract override"}
    probe = AllGraph(
        width=1,
        depth=1,
        epochs=1,
        device="cpu",
        verbose=False,
        seed=0,
        tensorize_mu=tzmu,
        contract_router=router,
        enabled_contracts=enabled_sg,
        equivariant_if_positions=True,
    )
    contract, detail = probe.route(train)
    if not probe._contract_enabled(contract):
        contract = probe._resolve_enabled_fallback(contract, train)
        detail = {**detail, "contract_restriction": f"restricted -> {contract}"}
    return contract, detail


def choose_budget(args, contract, train):
    """width/depth/epochs for a dataset nobody has tuned: start from that contract's benchmark budget, then
    let the CLI override any of the three. --epochs_scale still multiplies inside make_allgraph, so
    `--epochs 5 --epochs_scale 2.0` trains 10."""
    bud = dict(BUDGET.get(contract) or BUDGET["sequence"])
    if contract not in BUDGET:
        print(f"[budget] no BUDGET entry for contract {contract!r}; using the sequence budget")
    if contract == "sequence":
        # NB: for a flat table _seq_depth_for reads the FEATURE count, not a time axis. The receptive-field
        # argument still applies (a 600-column vector needs depth to see across itself), but the axis means
        # something different from what it means for a time series.
        bud["depth"] = _seq_depth_for(train, bud["depth"])
    for k in ("width", "depth", "epochs"):
        v = getattr(args, k)
        if v is not None:
            bud[k] = int(v)
    return bud


def row_name(args):
    if args.name:
        return args.name
    src = args.handle or os.path.basename(os.path.abspath(args.local_dir or "kaggle"))
    return re.sub(r"[^A-Za-z0-9._-]+", "-", src).strip("-") or "kaggle"


def ingest(root, buckets, args, mode):
    return {"tabular": encode_tabular, "images": encode_images, "npy": encode_arrays}[mode](root, buckets, args)


# --------------------------------------------------------------------------- inspect
def run_inspect(root, buckets, inv, args, tzmu, router, enabled_sg):
    """Print everything the runner would decide, and train nothing. This is the first thing to run against an
    unfamiliar handle: the file inventory, the detected mode, the per-column verdicts, the target and task
    guesses, the split, the chance baseline, the routed contract, and the budget.

    It reaches those answers by running the REAL ingest on a capped subset (--inspect_rows rows, 8 images per
    class), so what it prints is what the full run will do -- there is no second implementation to drift.
    """
    bar = "=" * 110
    print(bar)
    print(f"KAGGLE INSPECT  {args.handle or args.local_dir}   (nothing will be trained)")
    print(f"root  : {root}")
    print(f"bytes : {_human(sum(s for _, s in inv))} in {len(inv)} files")
    print("-" * 110)
    print(f"FILES (showing {min(len(inv), args.max_list)} of {len(inv)}, largest first)")
    for rel, size in inv[: args.max_list]:
        print(f"  {_human(size):>10}  {rel}")
    print("-" * 110)

    mode, why = detect_mode(root, buckets, args)
    if mode is None:
        print(f"MODE  = UNDETECTABLE -- {why}")
        print("        pass --mode tabular|images|npy together with --table / --image_root / --array")
        print(bar)
        return 2
    print(f"MODE  = {mode}    ({why})")
    print(
        f"        tables={len(buckets['tables'])} arrays={len(buckets['arrays'])} images={len(buckets['images'])}"
        f"   -- override with --mode / --table / --image_root / --array"
    )
    print("-" * 110)

    if mode == "tabular":
        try:
            path, rel = pick_table(root, buckets, args)
            df = read_table(path, args, nrows=min(args.inspect_rows, args.max_rows))
            target, t_why = guess_target(df, args)
            print(
                f"TABLE  {rel}   profiled {len(df)} rows (--inspect_rows {args.inspect_rows}), {len(df.columns)} cols"
            )
            print(f"  {'#':>3}  {'column':<26} {'dtype':<10} {'card':>7} {'miss':>6}  {'samples':<26} verdict")
            for i, r in enumerate(profile_table(df, target, args)):
                print(
                    f"  {i:>3}  {r['name'][:26]:<26} {r['dtype'][:10]:<10} {r['cardinality']:>7} "
                    f"{r['missing']:>6}  {r['samples'][:26]:<26} {r['verdict']}"
                )
            print(f"TARGET '{target}'   why: {t_why}")
        except Exception as e:
            print(f"TABLE  could not profile: {type(e).__name__}: {e}")
            print(bar)
            return 2
        print("-" * 110)

    capped = argparse.Namespace(**vars(args))
    capped.max_rows = min(args.max_rows, args.inspect_rows)
    capped.per_class = min(args.per_class, 8) if args.per_class else 8
    try:
        d = ingest(root, buckets, capped, mode)
    except Exception as e:
        print(f"INGEST would fail: {type(e).__name__}: {e}")
        print(bar)
        return 2

    m = d["meta"]
    print(f"TASK   {d['task']}   ({m.get('task_why', 'n/a')})")
    if m.get("class_names"):
        shown = ", ".join(m["class_names"][:12]) + (" ..." if len(m["class_names"]) > 12 else "")
        print(f"CLASSES {m['n_classes']}: {shown}   (train majority: {m.get('majority_class')})")
    for c in m.get("dropped_columns", []):
        print(f"DROP   {c}")
    print(f"SHAPE  train={tuple(d['train'].dense.shape)}  test={tuple(d['test'].dense.shape)}")
    print(f"SPLIT  {m['n_train']} train / {m['n_test']} test   test_frac={args.test_frac} split_seed={args.split_seed}")
    basis = "train-majority class scored on test" if d["task"] == "classification" else "R2 of the mean predictor"
    print(f"CHANCE {d['chance']:.4f}   ({basis})")
    if d.get("target_scale"):
        print(f"TARGET z-scored on train stats; MAE reported in [{d['target_units']}] (scale {d['target_scale']:.4g})")
    contract, detail = probe_contract(d["train"], args, tzmu, router, enabled_sg)
    bud = choose_budget(args, contract, d["train"])
    print(f"ROUTE  {contract}   {detail}")
    print(
        f"BUDGET width={bud['width']} depth={bud['depth']} epochs={bud['epochs']} (x --epochs_scale {args.epochs_scale})"
    )
    print(bar)
    print("Nothing was trained, and the above used a capped subset. Re-run without --inspect.")
    return 0


# --------------------------------------------------------------------------- CLI
def add_kaggle_args(ap):
    """The DATA-selection flags this runner owns. Everything about the MODEL comes from the standard runner's
    add_pipeline_args, so the two are configured identically."""
    g = ap.add_argument_group("acquisition")
    g.add_argument("--handle", default=None, help="Kaggle handle, e.g. uciml/iris (a competition is a BARE slug)")
    g.add_argument("--source", default="dataset", choices=sorted(_HANDLE_RE), help="which Kaggle resource type")
    g.add_argument("--local_dir", default=None, help="use an already-extracted directory; skips kagglehub entirely")
    g.add_argument("--file", default=None, help="download only this file from the resource (kagglehub path=)")
    g.add_argument("--force_download", action="store_true", help="re-download even when the cache has it")
    g.add_argument(
        "--cache_dir", default=None, help='KAGGLEHUB_CACHE; default <data-dir>/kagglehub ("" = kagglehub default)'
    )
    g.add_argument("--check_auth", action="store_true", help="print the resolved Kaggle identity and exit")

    g = ap.add_argument_group("inspect")
    g.add_argument(
        "--inspect", action="store_true", help="print inventory/columns/target/route and exit; trains nothing"
    )
    g.add_argument("--inspect_rows", type=int, default=5000, help="row cap for the --inspect profile")
    g.add_argument("--max_list", type=int, default=200, help="max files listed by --inspect")

    g = ap.add_argument_group("ingest (all modes)")
    g.add_argument("--mode", default="auto", choices=["auto", "tabular", "images", "npy"])
    g.add_argument("--name", default=None, help="results-row name (default: a slug of the handle)")
    g.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    g.add_argument("--test_frac", type=float, default=0.2)
    g.add_argument("--split_seed", type=int, default=0, help="seeds the train/test split and every subsample")
    g.add_argument(
        "--seed",
        type=int,
        default=0,
        help="model seed; applied to the AllGraph AFTER construction, since make_allgraph itself passes seed=0",
    )
    g.add_argument(
        "--kind_hint",
        default=None,
        choices=["sequence", "spatial", "volumetric", "4d"],
        help="force the dense contract, bypassing rank/mode routing",
    )
    g.add_argument("--no_standardize", action="store_true", help="skip feature standardization")
    g.add_argument("--max_mb", type=float, default=4096.0, help="abort if the decoded image tensor would exceed this")
    g.add_argument("--field", default=None, help="free-text field label recorded in the results row")
    g.add_argument("--sota", default=None, help="free-text reference number recorded in the results row")

    g = ap.add_argument_group("tabular")
    g.add_argument("--table", default=None, help="which table to train on (relpath or basename)")
    g.add_argument("--sep", default=None, help="column separator (default: sniffed)")
    g.add_argument("--encoding", default=None, help="file encoding (default: utf-8, falling back to latin-1)")
    g.add_argument("--target", default=None, help="target column (default: a conventional name, else the last column)")
    g.add_argument("--drop_cols", default="", help="comma-separated columns to drop")
    g.add_argument("--max_rows", type=int, default=200000)
    g.add_argument("--max_features", type=int, default=512, help="encoded width cap; widest one-hots drop first")
    g.add_argument("--max_classes", type=int, default=50)
    g.add_argument("--max_cardinality", type=int, default=20, help="categoricals above this are dropped")
    g.add_argument("--max_missing", type=float, default=0.5, help="drop columns missing more than this fraction")
    g.add_argument("--target_units", default=None, help="units label for the regression MAE")

    g = ap.add_argument_group("images")
    g.add_argument("--image_root", default=None, help="directory whose subdirectories are the classes")
    g.add_argument("--hw", type=int, default=32, help="resize to hw x hw (square: the spatial schema takes one hw)")
    g.add_argument("--gray", action="store_true", help="load as 1-channel greyscale instead of RGB")
    g.add_argument("--per_class", type=int, default=500, help="max images per class (0 = all)")
    g.add_argument("--labels_csv", default=None, help="labels table for a flat image directory")
    g.add_argument("--image_col", default=None, help="column of --labels_csv holding the image filename")
    g.add_argument("--label_col", default=None, help="column of --labels_csv holding the class label")
    g.add_argument(
        "--use_dir_split",
        default="auto",
        choices=["auto", "off"],
        help="auto: use a shipped train/test directory split; off: ignore it, re-split the largest class root",
    )
    g.add_argument("--min_images", type=int, default=32, help="images needed before auto-detection picks image mode")

    g = ap.add_argument_group("arrays")
    g.add_argument("--array", default=None, help="which .npy/.npz to use")
    g.add_argument("--x_key", default=None, help="feature array key inside a .npz")
    g.add_argument("--y_key", default=None, help="target array key inside a .npz")
    g.add_argument("--y_file", default=None, help="labels .npy sitting beside a features .npy")
    g.add_argument("--normalize", default="auto", choices=["auto", "none", "zscore", "unit"])

    g = ap.add_argument_group("budget / results")
    g.add_argument("--width", type=int, default=None, help="override the contract budget's width")
    g.add_argument("--depth", type=int, default=None, help="override the contract budget's depth")
    g.add_argument(
        "--epochs", type=int, default=None, help="override the budget's epochs (--epochs_scale still multiplies)"
    )
    g.add_argument(
        "--budget_contract", default=None, choices=sorted(BUDGET), help="skip the route probe, use this budget"
    )
    g.add_argument(
        "--results_out", default=None, help="default <data-dir>/kaggle_val_rows.json (NEVER the benchmark file)"
    )
    g.add_argument("--results_reset", action="store_true", help="start a fresh results document")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Fetch an arbitrary Kaggle dataset, route it, train an AllGraph, and score it.",
    )
    add_kaggle_args(ap)
    add_pipeline_args(ap)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.drop_set = {c.strip() for c in args.drop_cols.split(",") if c.strip()}
    if args.per_class == 0:
        args.per_class = None

    if args.check_auth:
        return _check_auth()
    root = resolve_source(args)
    if root is None:
        return 2
    inv = walk_inventory(root)
    if not inv:
        return _fail(f"{root} contains no files")
    buckets = classify_files(inv)
    # Resolve the pipeline before --inspect so the preview can show the REAL routed contract and budget. This
    # is pure argument/device/preset resolution and costs nothing.
    device, router, tzmu, enabled_sg = resolve_pipeline(args, ap)
    if args.inspect:
        return run_inspect(root, buckets, inv, args, tzmu, router, enabled_sg)

    mode, why = detect_mode(root, buckets, args)
    if mode is None:
        return _fail(
            f"could not tell what kind of dataset this is -- {why}. Largest files: {[r for r, _ in inv[:10]]}. "
            f"Pass --mode tabular|images|npy together with --table / --image_root / --array."
        )
    print(f"[mode] {mode}  ({why})")

    name = row_name(args)
    results_path = args.results_out or cache_path("kaggle_val_rows.json")
    doc = {"meta": {"runs": []}, "rows": {}} if args.results_reset else load_results(results_path)
    try:
        kh_ver = getattr(_kagglehub(), "__version__", None)
    except ImportError:
        kh_ver = None
    doc["meta"].setdefault("runs", []).append(
        {
            "command": " ".join(sys.argv),
            "device": str(device),
            "started": _now(),
            "git_sha": _git_sha(),
            "ilmarinen_version": getattr(ilmarinen, "__version__", None),
            "torch_version": torch.__version__,
            "kagglehub_version": kh_ver,
        }
    )

    # INGEST in its own guard: a bad column, an unreadable image or a missing key is a recorded OUTCOME for
    # this dataset, not a traceback -- the same discipline the standard runner applies to a failing loader.
    try:
        d = ingest(root, buckets, args, mode)
    except Exception as e:
        print(f"\n[{name}] INGEST FAILED -- {type(e).__name__}: {e}\n")
        record_row(doc, results_path, name, _stub_row(name, None, "error", f"ingest: {type(e).__name__}: {e}"))
        return 1
    d["meta"].update({"handle": args.handle, "source": args.source, "root": root, "mode": mode, "mode_reason": why})

    t0, mg = time.time(), None
    try:
        # Order matters: probe -> budget -> flatten. maybe_flatten_grids mutates `d` in place, so probing
        # after it would size the budget from a flattened vector instead of from the real contract.
        expected, route_detail = probe_contract(d["train"], args, tzmu, router, enabled_sg)
        run_args = apply_opt_preset(args, expected)
        bud = choose_budget(run_args, expected, d["train"])
        maybe_flatten_grids(run_args, d, expected)
        print(f"[route] {expected}  {route_detail}")
        print(f"[budget] width={bud['width']} depth={bud['depth']} epochs={bud['epochs']} device={device}")

        mg = make_allgraph(run_args, bud, device, router, tzmu, enabled_sg)
        mg.seed = int(args.seed)  # make_allgraph passes a literal seed=0; every consumer reads self.seed at fit
        mg.progress_desc = name
        res = mg.fit(
            d["train"],
            task=d["task"],
            n_out=d["meta"].get("n_classes"),
            select=run_args.select,
            tiebreak=run_args.tiebreak,
            select_size=run_args.select_size,
        )
        metric, value, extra = _eval_test(
            mg, d["test"], d["task"], target_scale=d.get("target_scale"), target_units=d.get("target_units")
        )
        chance = float(d["chance"])
        extra_d = dict(extra)
        if d["task"] == "regression":
            skill = extra_d.get("R2", value)
        elif chance >= 1.0:
            # A degenerate single-class test split makes the normalization undefined. No registry dataset can
            # hit this; an arbitrary Kaggle target absolutely can.
            skill = float("nan")
        else:
            skill = (value - chance) / (1.0 - chance)
        dt = time.time() - t0
        arch = (
            "→".join(res.get("architecture") or [c.primitives[int(c.alpha.argmax())] for c in mg.net.cells])
            if hasattr(mg.net, "cells")
            else "?"
        )
        params = sum(p.numel() for p in mg.net.parameters())
        extra_str = "".join(f" {n}={v:.4f}" for n, v in extra)
        tag = f" IPR={res['ipr']:.2f}" if "ipr" in res else ""
        print("=" * 110)
        print(
            f"[{mg.contract:11}] {name:24} {metric:12}={value:.4f}{extra_str} skill={skill:+.3f} "
            f"arch=[{arch}]{tag} params={params:>8} {dt:.0f}s"
        )
        print(f"{'':13} {'':24} chance={chance:.4f}   route={mg.route_detail}")
        saved, save_error = None, None
        if args.save_models:
            try:
                saved = mg.save(stem=name)
                print(f"{'':13} {'':24} saved model -> {saved}")
            except Exception as se:  # a save failure must never cost us the measurement
                save_error = f"{type(se).__name__}: {se}"
                print(f"{'':13} {'':24} SAVE FAILED (result kept) -- {save_error[:70]}")
        if res.get("converged") is False:
            print(f"{'':13} {'':24} NOT CONVERGED -- hit the {res.get('epoch_cap')}-epoch ceiling")
        record_row(
            doc,
            results_path,
            name,
            {
                "name": name,
                "status": "ok",
                "contract": mg.contract,
                "expected_contract": expected,
                "task": d["task"],
                "metric": metric,
                "value": float(value),
                "extra": {n: float(v) for n, v in extra},
                "skill": float(skill),
                "chance": chance,
                "arch": arch,
                "ipr": float(res["ipr"]) if "ipr" in res else None,
                "params": int(params),
                "width": int(mg.width),
                "depth": int(mg.depth),
                "epochs_trained": res.get("epochs_trained"),
                "converged": res.get("converged"),
                "epoch_cap": res.get("epoch_cap"),
                "auto_epoch_monitor": res.get("auto_epoch_monitor"),
                "field": d["field"],
                "sota": d["sota"],
                "seconds": round(dt, 1),
                "saved_model": saved,
                "save_error": save_error,
                # Everything needed to reconstruct WHAT was trained on. An arbitrary dataset's row has to be
                # self-describing: unlike a registry row, there is no loader to look the details up in.
                "kaggle": {**d["meta"], "route_detail": str(route_detail), "kagglehub_version": kh_ver},
            },
        )
        print(f"results -> {results_path}")
        print("=" * 110)
        return 0
    except Exception as e:
        print(f"\n[{name}] FAILED -- {type(e).__name__}: {e}\n")
        record_row(doc, results_path, name, _stub_row(name, None, "error", f"{type(e).__name__}: {e}"))
        return 1
    finally:
        mg = None
        d = None
        gc.collect()
        if str(device).startswith("mps") and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
