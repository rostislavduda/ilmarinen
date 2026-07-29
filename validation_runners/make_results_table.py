#!/usr/bin/env python
"""
make_results_table.py -- render the standard-validation results JSON as the README's markdown table.

run_standard_validation.py accumulates one merge-on-write document (default
<data-dir>/standard_val_rows.json) across however many --contracts / --only batches the suite was run
in. This turns that document into a table grouped by computational contract, so the README claim is
GENERATED from the recorded numbers rather than transcribed by hand.

USAGE:
    python -m validation_runners.make_results_table                 # print to stdout
    python -m validation_runners.make_results_table --out t.md      # write to a file
    python -m validation_runners.make_results_table --insert-readme  # splice into README.md

--insert-readme rewrites only the region between the marker comments

    <!-- BEGIN:stdval --> ... <!-- END:stdval -->

so the surrounding prose is never touched and the table stays regenerable after further batches.
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


def insert_readme(readme_path, table):
    """Replace the region between the marker comments, leaving the surrounding prose untouched."""
    with open(readme_path) as fh:
        text = fh.read()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme_path} has no {BEGIN} / {END} markers -- add them around the table region first.")
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    new = f"{pre}{BEGIN}\n\n{table}\n{END}{post}"
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
        help="results JSON written by run_standard_validation (default: <data-dir>/standard_val_rows.json)",
    )
    ap.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    ap.add_argument(
        "--insert-readme",
        nargs="?",
        const="README.md",
        default=None,
        metavar="PATH",
        help=f"splice the table into PATH (default README.md) between the {BEGIN} / {END} markers",
    )
    ap.add_argument("--no-failed", action="store_true", help="omit the trailing list of skipped/errored datasets")
    args = ap.parse_args()

    path = args.results or cache_path("standard_val_rows.json")
    if not os.path.exists(path):
        raise SystemExit(f"no results at {path} -- run validation_runners.run_standard_validation first")
    with open(path) as fh:
        doc = json.load(fh)
    table = render(doc, include_failed=not args.no_failed)

    if args.insert_readme:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = args.insert_readme
        insert_readme(target if os.path.isabs(target) else os.path.join(root, target), table)
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(table)
        print(f"wrote {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
