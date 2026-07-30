#!/usr/bin/env python
"""
make_results_table.py -- render a validation results JSON as one of the README's markdown tables.

Two documents, two tables, one renderer:

  DEFAULT  run_standard_validation.py accumulates one merge-on-write document (default
           <data-dir>/standard_val_rows.json) across however many --contracts / --only batches the
           suite was run in. This turns that document into a table grouped by computational contract.

  --kaggle run_kaggle_validation.py accumulates the same row schema for ARBITRARY Kaggle datasets in a
           SEPARATE document (default <data-dir>/kaggle_val_rows.json). Those rows are deliberately
           kept out of the benchmark table -- a user-chosen dataset with hand-picked settings is not a
           benchmark entry -- so they render into their own region with their own columns (ingest mode,
           chance baseline, data shape) and no SOTA column, since there is no reference number.

Either way the README claim is GENERATED from the recorded numbers rather than transcribed by hand.

USAGE:
    python -m validation_runners.make_results_table                    # print to stdout
    python -m validation_runners.make_results_table --out t.md         # write to a file
    python -m validation_runners.make_results_table --insert-readme    # splice into README.md
    python -m validation_runners.make_results_table --kaggle --insert-readme

--insert-readme rewrites only the region between the marker comments

    <!-- BEGIN:stdval  --> ... <!-- END:stdval  -->      (default)
    <!-- BEGIN:kaggleval --> ... <!-- END:kaggleval -->   (--kaggle)

so the surrounding prose is never touched and the table stays regenerable after further runs.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ilmarinen.core.paths import cache_path

BEGIN = "<!-- BEGIN:stdval -->"
END = "<!-- END:stdval -->"
KAGGLE_BEGIN = "<!-- BEGIN:kaggleval -->"
KAGGLE_END = "<!-- END:kaggleval -->"

# Contract order for the table: the dense/grid family first (sequence -> spatial -> volumetric -> 4d),
# then the relational family, then set and operator -- the same ordering the package docstring uses.
CONTRACT_ORDER = ["sequence", "spatial", "volumetric", "4d", "graph", "equivariant", "set", "operator"]

# Per-dataset caveats, rendered as footnotes. These flag rows where the registry's SOTA string and the
# loader's target are NOT measuring the same quantity -- a gap the numbers alone would misrepresent.
DATASET_CAVEATS = {
    "QM9": (
        "the loader regresses raw U0 TOTAL energy (z-scored, rescaled to meV), whereas the quoted "
        "~5-15 meV literature figure is for ATOMIZATION energy -- the standard QM9 target, obtained "
        "after subtracting atomic reference energies. The two are not comparable; read the R2 instead."
    ),
}

CONTRACT_BLURB = {
    "sequence": "1D series (UCR/UEA, epidemiological, tabular)",
    "spatial": "2D grids (images)",
    "volumetric": "3D grids (medical volumes)",
    "4d": "3D+time grids (solver-generated fields)",
    "graph": "molecular and social graphs",
    "equivariant": "E(3)/SO(3) point clouds (quantum chemistry, shapes)",
    "set": "permutation-invariant sets (particle physics)",
    "operator": "function -> function on a grid (PDE surrogates)",
    "generated_equivariant": (
        "a contract GENERATED for a group the symmetry front-end discovered, rather than one of the eight "
        "built-ins (reached under `--discover extended`)"
    ),
}


def fmt_arch(row):
    """The selected architecture.

    Discovered-group contracts deploy an EMLP built from the group's generators, not a stack of alpha
    cells, so the per-layer primitive readout is empty and the runner records `?`. Name the group instead
    -- on that path the group IS the architecture."""
    arch = row.get("arch") or "?"
    if arch != "?":
        return f"`{arch}`"
    if row.get("contract") == "generated_equivariant":
        g, n = row.get("group"), row.get("group_generators")
        if g:
            return f"EMLP `{g}`" + (f" ({n} generators)" if n else "")
        if n:
            return f"EMLP (discovered group, {n} generators)"
        return "EMLP (discovered group)"
    return "`?`"


def _split_top_level(s, sep=";"):
    """Split on `sep` only at parenthesis depth 0. Several registry SOTA strings carry a semicolon INSIDE
    their parenthetical (e.g. "R2 n/a (synthetic-from-solver; -> 1 for a sufficient 4d model)"), and a naive
    split severs the parenthesis mid-way."""
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def trim_sota(s):
    """Compress a registry SOTA string to one table cell.

    The registry stores free prose (`"MAE ~0.40-0.45 log mol/L (D-MPNN/best GNN, random split); R2
    ~0.90-0.93"`) because it has to describe several metrics at once. For the table we keep the FIRST
    clause -- the headline figure, with its parenthetical naming the method. The full string stays in the
    JSON, so nothing is lost; this only decides what fits in a column.
    """
    if not s:
        return "n/a"
    head = _split_top_level(s)[0]
    # a leading "acc "/"R2 "/"field R2 " label duplicates the table's own Metric column -- drop it
    head = re.sub(r"^(acc|R2|ROC-AUC|field R2|regression R2|energy MAE|MAE|U0 MAE)\s+", "", head)
    return head or "n/a"


def fmt_value(row):
    """The headline metric, plus any secondary figure the dataset reports (AUC, physical-unit MAE).

    Four decimals, not three: several SOTA references sit at ~0.999, and at 3dp a 0.9996 field-R2 renders
    as a flat "1.000" -- overstating the result and erasing the gap the column exists to show."""
    v = f"{row['value']:.4f}"
    extra = row.get("extra") or {}
    # rejection figures are O(100-3000) -> :.4g; scale-bound metrics keep the fixed 4 decimals
    bits = [f"{n} {x:.4g}" if n.startswith("1/eB") else f"{n} {x:.4f}" for n, x in sorted(extra.items())]
    return v + (f" <br><sub>{', '.join(bits)}</sub>" if bits else "")


def fmt_metric(row):
    """The metric name, marked with the direction of improvement.

    Most rows are higher-is-better (acc / R2 / AUC), but the chemistry datasets report a physical-unit
    ERROR (`MAE[log mol/L]`, `MAE[kcal/mol]`). Without a marker, a reader comparing a 0.66 MAE against a
    '~0.40-0.45' SOTA reference reads the gap backwards."""
    m = row.get("metric", "?")
    lower_better = "MAE" in m or "MSE" in m or "RMSE" in m
    return f"{m} ↓" if lower_better else m


def fmt_epochs(row):
    """Epochs trained, flagged when the ceiling -- not convergence -- ended training."""
    ep = row.get("epochs_trained")
    if ep is None:
        return "-"
    return f"{ep}" if row.get("converged") else f"{ep}†"


def render(doc, include_failed=True):
    rows = doc.get("rows", {})
    ok = {n: r for n, r in rows.items() if r.get("status") == "ok"}
    failed = {n: r for n, r in rows.items() if r.get("status") != "ok"}

    out = []
    by_contract = {}
    for n, r in ok.items():
        by_contract.setdefault(r.get("contract") or r.get("expected_contract"), {})[n] = r

    # contracts in the canonical order, then any the model routed to that isn't in it (e.g.
    # generated_equivariant, which only appears when --discover finds a group)
    ordered = [c for c in CONTRACT_ORDER if c in by_contract]
    ordered += [c for c in sorted(by_contract) if c not in CONTRACT_ORDER]

    any_unconverged = any(r.get("converged") is False for r in ok.values())

    for contract in ordered:
        blurb = CONTRACT_BLURB.get(contract)
        out.append(f"**`{contract}`**" + (f" -- {blurb}" if blurb else ""))
        out.append("")
        out.append("| dataset | metric | ilmarinen | SOTA reference | skill | architecture | params | epochs |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name in sorted(by_contract[contract], key=lambda n: -by_contract[contract][n].get("skill", 0)):
            r = by_contract[contract][name]
            label = f"{name} ‡" if name in DATASET_CAVEATS else name
            out.append(
                f"| {label} | {fmt_metric(r)} | {fmt_value(r)} | {trim_sota(r.get('sota'))} | "
                f"{r.get('skill', float('nan')):+.4f} | {fmt_arch(r)} | "
                f"{r.get('params', 0):,} | {fmt_epochs(r)} |"
            )
        out.append("")

    notes = []
    if any(("MAE" in (r.get("metric") or "") or "MSE" in (r.get("metric") or "")) for r in ok.values()):
        notes.append("↓ lower is better (a physical-unit error); every other metric is higher-is-better.")
    for ds in sorted(n for n in ok if n in DATASET_CAVEATS):
        notes.append(f"‡ **{ds}** -- {DATASET_CAVEATS[ds]}")
    if any_unconverged:
        notes.append(
            "† training stopped at the epoch ceiling rather than on the convergence criterion -- "
            "these numbers are budget-limited, not converged."
        )
    if notes:
        out.extend(notes + [""])

    if include_failed and failed:
        out.append("Not measured in this run:")
        out.append("")
        for name in sorted(failed):
            r = failed[name]
            out.append(f"- **{name}** ({r.get('expected_contract', '?')}) -- {r.get('status')}: {r.get('note', '')}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- the Kaggle (bring-your-own) table
_MODE_BLURB = {
    "tabular": "a CSV/parquet table -> numeric + one-hot matrix (rank 2)",
    "images": "an image class-directory tree -> (N, C, hw, hw) (rank 4)",
    "npy": "a raw .npy/.npz array -> rank passed through to the grid-rank router",
}


def kaggle_link(row):
    """The dataset name, linked to its Kaggle page when the row records a handle. --local_dir runs have no
    handle (they never touched Kaggle), so those render as a plain name."""
    name = row.get("name", "?")
    k = row.get("kaggle") or {}
    handle, source = k.get("handle"), k.get("source", "dataset")
    if not handle:
        return name
    base = "competitions" if source == "competition" else "datasets"
    return f"[{name}](https://www.kaggle.com/{base}/{handle})"


def kaggle_shape(row):
    """A compact data-shape cell: per-sample shape where it is recoverable, else the feature count.

    Images record hw + gray, so the (C, hw, hw) tensor the model actually saw can be reconstructed exactly;
    a table only has a width. Written defensively because these rows come from ad-hoc runs whose flags vary.
    """
    k = row.get("kaggle") or {}
    n_tr, n_te = k.get("n_train"), k.get("n_test")
    counts = f"{n_tr:,}/{n_te:,}" if isinstance(n_tr, int) and isinstance(n_te, int) else "?"
    hw, feats = k.get("hw"), k.get("n_features")
    if hw:
        shape = f"{1 if k.get('gray') else 3}x{hw}x{hw}"
    elif isinstance(feats, int):
        shape = f"{feats} feat"
    else:
        shape = "?"
    return f"{counts}<br><sub>{shape}</sub>"


def kaggle_epochs(row):
    """Epochs trained -- THREE states, not the standard table's two.

    `converged` is None when no --auto_epoch criterion was applied at all, which is the default for an
    ad-hoc run and never happens in the benchmark suite (it always passes --auto_epoch val). Such a row
    trained exactly its budget and convergence was never TESTED, so reusing the standard table's `†`
    ("stopped at the ceiling without converging") would assert something that was not measured.
    """
    ep = row.get("epochs_trained")
    if ep is None:
        return "-"
    conv = row.get("converged")
    if conv is True:
        return f"{ep}"
    return f"{ep}†" if conv is False else f"{ep}*"


def kaggle_settings(row):
    """The run-defining choices, so a number in the table can be reproduced rather than merely believed.

    Deliberately the DATASET-SPECIFIC part only, rendered the same way for every row. The shared pipeline
    flags are identical across these runs and are stated once in the surrounding prose, so repeating the
    full ~20-flag command per row would bury the one or two things that actually differ (channel count,
    resolution, target column). The exact invoking command is recorded per row in the results JSON
    (`kaggle.command`) for byte-exact reproduction, so nothing is lost by not printing it here.

    Everything shown is READ from the row rather than assumed: an `auto_epoch_monitor` means --auto_epoch
    ran, an `ipr` means the sparsity-priced mixture ran, and width/depth are what the size selector actually
    settled on, not what was requested.
    """
    k = row.get("kaggle") or {}
    bits = [f"`--mode {k.get('mode', '?')}`"]
    if k.get("mode") == "tabular" and k.get("target"):
        bits.append(f"`--target {k['target']}`")
    if k.get("hw"):
        bits.append(f"`--hw {k['hw']}`" + (" `--gray`" if k.get("gray") else ""))
    if k.get("per_class"):
        bits.append(f"`--per_class {k['per_class']}`")
    elif k.get("mode") == "images":
        bits.append("`--per_class 0` (all images)")
    if row.get("auto_epoch_monitor"):
        bits.append(f"`--auto_epoch {row['auto_epoch_monitor']}`")
    if row.get("ipr") is not None:
        bits.append("`--select sparse`")
    if row.get("width") and row.get("depth"):
        bits.append(f"selected width {row['width']}, depth {row['depth']}")
    # The ceiling and what actually happened are different facts: reporting "1000-epoch budget" for a run
    # that converged at 29 reads as though it trained 1000.
    ep, cap = row.get("epochs_trained"), row.get("epoch_cap")
    if ep is not None and cap and cap != ep:
        bits.append(f"stopped at {ep} of a {cap}-epoch ceiling")
    elif cap:
        bits.append(f"{cap}-epoch budget")
    if k.get("split"):
        bits.append(k["split"])
    return ", ".join(bits)


def render_kaggle(doc, include_failed=True):
    """One flat table -- these are individually-chosen datasets, not a suite, so grouping by contract would
    put one row under each heading. The ingest mode and the routed contract are columns instead."""
    rows = doc.get("rows", {})
    ok = {n: r for n, r in rows.items() if r.get("status") == "ok"}
    failed = {n: r for n, r in rows.items() if r.get("status") != "ok"}

    out = []
    if not ok:
        out.append("_No Kaggle datasets recorded yet._")
        out.append("")
    else:
        out.append(
            "| dataset | task | ingest -> contract | ilmarinen | chance | skill | architecture | params | epochs | train/test |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for name in sorted(ok, key=lambda n: -(ok[n].get("skill") or 0)):
            r = ok[name]
            k = r.get("kaggle") or {}
            chance = r.get("chance")
            out.append(
                f"| {kaggle_link(r)} | {r.get('task', '?')} | `{k.get('mode', '?')}` -> `{r.get('contract', '?')}` | "
                f"{fmt_value(r)} | {chance:.4f} | {r.get('skill', float('nan')):+.4f} | {fmt_arch(r)} | "
                f"{r.get('params', 0):,} | {kaggle_epochs(r)} | {kaggle_shape(r)} |"
            )
        out.append("")
        out.append("Settings per run (everything else is the shared pipeline default):")
        out.append("")
        for name in sorted(ok):
            out.append(f"- **{name}** -- {kaggle_settings(ok[name])}")
        out.append("")
        if any(r.get("converged") is False for r in ok.values()):
            out.append(
                "† training stopped at the epoch ceiling rather than on the convergence criterion -- that "
                "number is budget-limited, not converged."
            )
            out.append("")
        if any(r.get("converged") is None for r in ok.values()):
            out.append(
                "\\* trained a fixed epoch budget with no convergence criterion (`--auto_epoch` was not "
                "set), so whether the model had converged was not tested -- the number may be either "
                "under- or over-trained."
            )
            out.append("")

    if include_failed and failed:
        out.append("Attempted and not measured:")
        out.append("")
        for name in sorted(failed):
            out.append(f"- **{name}** -- {failed[name].get('status')}: {failed[name].get('note', '')}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def insert_readme(readme_path, table, begin=BEGIN, end=END):
    """Replace the region between the marker comments, leaving the surrounding prose untouched."""
    with open(readme_path) as fh:
        text = fh.read()
    if begin not in text or end not in text:
        raise SystemExit(f"{readme_path} has no {begin} / {end} markers -- add them around the table region first.")
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    new = f"{pre}{begin}\n\n{table}\n{end}{post}"
    if new == text:
        print(f"{readme_path}: table already current")
        return
    with open(readme_path, "w") as fh:
        fh.write(new)
    print(f"{readme_path}: table updated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results",
        default=None,
        help="results JSON to render (default: <data-dir>/standard_val_rows.json, or kaggle_val_rows.json with --kaggle)",
    )
    ap.add_argument(
        "--kaggle",
        action="store_true",
        help="render the bring-your-own-dataset table from run_kaggle_validation's SEPARATE results document, "
        f"into the {KAGGLE_BEGIN} / {KAGGLE_END} region",
    )
    ap.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    ap.add_argument(
        "--insert-readme",
        nargs="?",
        const="README.md",
        default=None,
        metavar="PATH",
        help="splice the table into PATH (default README.md) between the marker comments for the chosen table",
    )
    ap.add_argument("--no-failed", action="store_true", help="omit the trailing list of skipped/errored datasets")
    args = ap.parse_args()

    default_name = "kaggle_val_rows.json" if args.kaggle else "standard_val_rows.json"
    runner = "run_kaggle_validation" if args.kaggle else "run_standard_validation"
    path = args.results or cache_path(default_name)
    if not os.path.exists(path):
        raise SystemExit(f"no results at {path} -- run validation_runners.{runner} first")
    with open(path) as fh:
        doc = json.load(fh)
    renderer = render_kaggle if args.kaggle else render
    table = renderer(doc, include_failed=not args.no_failed)

    if args.insert_readme:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = args.insert_readme
        begin, end = (KAGGLE_BEGIN, KAGGLE_END) if args.kaggle else (BEGIN, END)
        insert_readme(target if os.path.isabs(target) else os.path.join(root, target), table, begin=begin, end=end)
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(table)
        print(f"wrote {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
