"""record_contract_corpus.py -- build a learned-contract-router corpus from FULL-budget bake-off outcomes.

The learned contract router (machinery/learned_contract_router.py) is only as good as the bake-off labels
it learns from. Reduced-budget bake-offs can mislabel: at a tiny tie-break budget the equivariant branch
has not converged, so a geometric dataset can spuriously appear to prefer graph/set. This utility runs the
bake-off at an ADEQUATE budget on every geometric dataset (positions present) in the registry, records the
(descriptor, full-budget-winner) pairs into a ContractRouter, and persists the corpus to JSON so the
shipped default_router can be upgraded from real outcomes rather than hand-seeded archetypes.

Usage:
    python validation_runners/record_contract_corpus.py [--epochs 90] [--out corpus.json] [--reduced]

The premise for doing this (verified in tests/learned_contract_routing.md): the bake-off winner on the
geometric datasets FLIPS toward equivariant as budget grows (rMD17 R2 0.06->0.44, QM7 0.62->0.72 for
equivariant while set/graph stay flat), so full-budget labels are materially better than reduced ones.
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from ilmarinen.core.allgraph import AllGraph
from ilmarinen.core.dataset_registry import full_suite, quick_suite
from ilmarinen.machinery import ContractRouter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90, help="bake-off budget per candidate contract")
    from ilmarinen.core.paths import cache_path
    ap.add_argument("--out", default=cache_path("contract_corpus.json"),
                    help="where to persist the corpus JSON (default: <cache>/contract_corpus.json)")
    ap.add_argument("--reduced", action="store_true", help="use reduced dataset subsets (faster)")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    args = ap.parse_args()

    suite = full_suite() if not args.reduced else quick_suite()
    names = list(suite)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]

    router = ContractRouter(min_confidence=0.4)
    print("=" * 90)
    print(f"RECORDING FULL-BUDGET BAKE-OFF CORPUS  (epochs={args.epochs}, reduced={args.reduced})")
    print("Only geometric datasets (positions present) have an ambiguous contract -> a bake-off.")
    print("=" * 90)
    recorded = 0
    for name in names:
        loader, expected_mod, _ = suite[name]
        try:
            d = loader(reduced=args.reduced, device="cpu")
            tr = d["train"]
            if getattr(tr, "positions", None) is None:
                continue  # not geometric -> no ambiguity, structural dispatch is certain
            torch.manual_seed(0)
            # bake off with the learned router DISABLED so we record the raw full-budget verdict
            mg = AllGraph(width=16, depth=2, epochs=args.epochs, verbose=False, seed=0, contract_router=None)
            descriptor = mg._dataset_descriptor(tr)
            t0 = time.time()
            winner, scores, detail = mg.tiebreak(tr, task=d["task"], mu_c=0.05, tiebreak_epochs=args.epochs)
            router.add(descriptor, winner)
            recorded += 1
            sc = {k: round(v, 3) for k, v in scores.items()}
            print(f"[{name:16}] winner={winner:12} scores={sc}  desc={[round(float(x),2) for x in descriptor]}  ({time.time()-t0:.0f}s)")
        except FileNotFoundError:
            print(f"[{name:16}] SKIP -- data not present")
        except Exception as e:
            print(f"[{name:16}] ERROR -- {type(e).__name__}: {str(e)[:60]}")
    print("=" * 90)
    print(f"Recorded {recorded} geometric datasets into the corpus.")
    if recorded:
        with open(args.out, "w") as f:
            f.write(router.to_json())
        print(f"Corpus persisted to {args.out}. Load with ContractRouter.from_json(open(path).read()).")
        # sanity: leave-one-out over the recorded corpus
        if recorded >= 3:
            X, y = router.X, np.array(router.y)
            correct = 0
            for i in range(len(X)):
                m = np.arange(len(X)) != i
                sub = ContractRouter(); sub.X = X[m]; sub.y = list(y[m]); sub._fit()
                pred, conf, _ = sub.predict(X[i])
                correct += (pred == y[i])
            print(f"leave-one-out over recorded corpus: {correct}/{len(X)}")


if __name__ == "__main__":
    main()
