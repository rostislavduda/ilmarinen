"""Learned, transferable contract routing: amortize the per-dataset contract bake-off.

The meta-router's one genuinely ambiguous decision is the CONTRACT for geometric data (positions
present): equivariant vs graph vs set. The principled answer is a bake-off -- train each candidate
contract from scratch on a held-out split and pick argmin J = R_c + mu_c * Omega_struct(c) -- but that
is expensive (2-3 full trains per new dataset) and NON-TRANSFERABLE (nothing is learned across datasets).

This module amortizes the bake-off with a cheap LEARNED predictor: a small set of training-free dataset
descriptors -> the contract the bake-off would choose. It is a surrogate for argmin_c[R_c + mu_c*Omega]
that predicts the argmin WITHOUT computing each R_c; Omega_struct remains the tie-break prior, so this is
an amortized approximation of the SAME MDL objective, not a departure from it.

THE DESCRIPTORS (cheap, training-free, dimensionless for transfer). The discriminative signal is WHAT THE
TARGET DEPENDS ON, captured by three proxy-correlations of y with structure-specific descriptors:
  geo_proxy  = max_k |corr(pairwise-distance summary_k, y)|   (geometry; rotation-invariant)
  topo_proxy = max_k |corr(adjacency-spectrum + degree stat_k, y)|  (topology; geometry-blind)
  set_proxy  = max_k |corr(node-feature moment_k, y)|         (set; permutation & geometry & topology blind)
plus normalized structural stats (avg N, avg E). Validated: the winning proxy always matches the true
contract on controlled targets (geo 0.98/topo 0.20/set 0.08 for a geometric target, etc.), and a
prototype router transfers 12/12 leave-one-out across dataset scales (tests/learned_contract_routing.md).

THE MODEL. A nearest-prototype (class-centroid) classifier in standardized descriptor space -- transparent,
few-shot, and calibrated by the margin between the nearest and runner-up prototype (the confidence). A
gradient-boosted or logistic model could replace it, but the prototype model is defensible from a handful
of datasets and needs no tuning.

THE SAFETY POLICY (correctness floor). The learned router predicts (contract, confidence). If confident,
use the prediction and skip the bake-off. If not, FALL BACK to the bake-off (the ground truth) and RECORD
the outcome to improve the corpus. So the router is never WORSE than the bake-off (it defers when unsure)
and is much cheaper when confident. A warm default corpus ships so it is useful out of the box.
"""
from __future__ import annotations
import json
import numpy as np


# --------------------------------------------------------------------------- descriptor
def _per_molecule_blocks(positions, node_feats, edges_or_adj):
    """Return (geo, topo, set) per-molecule descriptor rows for a batch. Robust to missing pieces."""
    geo, topo, st = [], [], []
    for p, Z, A in zip(positions, node_feats, edges_or_adj):
        Z = np.asarray(Z, float).reshape(len(Z), -1)
        zc = Z.mean(axis=1) if Z.shape[1] > 1 else Z[:, 0]        # a scalar per node
        N = len(zc)
        # geometry: pairwise-distance summary (rotation invariant)
        if p is not None and N > 1:
            P = np.asarray(p, float)
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            du = D[np.triu_indices(N, 1)]
            geo.append([du.mean(), du.std(), du.max()])
        else:
            geo.append([0.0, 0.0, 0.0])
        # topology: adjacency spectrum + degree stats (geometry blind)
        if A is not None and N > 1:
            A = np.asarray(A, float)
            ev = np.sort(np.linalg.eigvalsh(A))[::-1]
            deg = A.sum(1)
            topo.append([ev[0], ev[1] if N > 1 else 0.0, deg.mean(), deg.std()])
        else:
            topo.append([0.0, 0.0, 0.0, 0.0])
        # set: node-feature moments (permutation & geometry & topology blind)
        st.append([zc.mean(), zc.std(), float(np.sort(zc)[-1]), float(np.median(zc))])
    return np.array(geo), np.array(topo), np.array(st)


def dataset_descriptor(positions, node_feats, adjacencies, y):
    """A single fixed-length, transfer-friendly descriptor vector for a whole dataset.

    positions   : list of (N_i, d) arrays or None per datum (None if no coordinates)
    node_feats  : list of (N_i, F) node feature arrays
    adjacencies : list of (N_i, N_i) dense adjacency arrays or None per datum
    y           : (n_data,) target values (used only via correlations -- the amortized "which structure
                  explains y" signal). If y is None or constant, proxy-correlations are set to 0.

    Returns a length-5 vector [geo_proxy, topo_proxy, set_proxy, avgN/10, avgE/20].
    """
    geo, topo, st = _per_molecule_blocks(positions, node_feats, adjacencies)
    y = None if y is None else np.asarray(y, float).ravel()

    def best_proxy_corr(P):
        if y is None or np.std(y) < 1e-9 or P.shape[0] != len(y):
            return 0.0
        cs = []
        for k in range(P.shape[1]):
            col = P[:, k]
            if np.std(col) < 1e-9:
                continue
            c = np.corrcoef(col, y)[0, 1]
            if np.isfinite(c):
                cs.append(abs(c))
        return max(cs) if cs else 0.0

    avg_N = np.mean([len(z) for z in node_feats]) / 10.0
    avg_E = np.mean([(np.asarray(a).sum() / 2.0 if a is not None else 0.0) for a in adjacencies]) / 20.0
    return np.array([best_proxy_corr(geo), best_proxy_corr(topo), best_proxy_corr(st),
                     float(avg_N), float(avg_E)], float)


# --------------------------------------------------------------------------- the router
class ContractRouter:
    """Nearest-prototype learned router over dataset descriptors, with a confidence margin and a
    persistable corpus. Predicts the contract the bake-off would choose; defers (low confidence) so the
    caller can fall back to the bake-off and record the true outcome."""

    def __init__(self, min_confidence=0.5, match_dims=3):
        self.X = np.zeros((0, 5))          # descriptor corpus (full 5-dim: 3 proxies + 2 size stats)
        self.y = []                        # contract labels
        self.min_confidence = min_confidence
        # Only the first `match_dims` descriptor features are used for prototype matching. The proxy-
        # correlations (dims 0-2: geo/topo/set) are dimensionless in [0,1] and TRANSFER across physical
        # domains; the raw size stats (dims 3-4: normalized N, E) do NOT -- a molecule's bond count and a
        # particle cloud's proximity-edge count live on wildly different scales, so including them in the
        # standardized distance swamps the discriminative proxies and breaks cross-domain transfer (a
        # measured failure: cross-domain leave-one-out 4/7 with all 5 dims vs 6/7 with the 3 proxies). The
        # full 5-dim descriptor is still stored (informative for inspection); matching uses match_dims.
        self.match_dims = match_dims
        self._mu = None
        self._sd = None
        self._protos = {}

    # ---- corpus management ----
    def add(self, descriptor, contract):
        """Record a (descriptor, bake-off-winner) pair and refit the prototypes."""
        self.X = np.vstack([self.X, np.asarray(descriptor, float).reshape(1, -1)])
        self.y.append(str(contract))
        self._fit()
        return self

    def _fit(self):
        if len(self.y) == 0:
            self._protos = {}
            return
        Xm = self.X[:, :self.match_dims]                 # transferable proxy features only
        self._mu = Xm.mean(0)
        self._sd = Xm.std(0) + 1e-9
        Xs = (Xm - self._mu) / self._sd
        ys = np.array(self.y)
        self._protos = {c: Xs[ys == c].mean(0) for c in sorted(set(self.y))}

    # ---- prediction ----
    def predict(self, descriptor):
        """Return (contract, confidence, detail). confidence is the standardized-distance margin between
        the nearest and runner-up prototype, squashed to [0,1). Returns (None, 0, ...) if untrained or
        only one class is known (cannot discriminate)."""
        if not self._protos or len(self._protos) < 2:
            return None, 0.0, {"reason": "router has fewer than two contract classes; defer to bake-off"}
        xs = (np.asarray(descriptor, float)[:self.match_dims] - self._mu) / self._sd
        dists = {c: float(np.linalg.norm(xs - p)) for c, p in self._protos.items()}
        order = sorted(dists, key=dists.get)
        best, runner = order[0], order[1]
        margin = dists[runner] - dists[best]
        confidence = margin / (margin + 1.0)          # monotone squash to [0,1)
        detail = {"distances": dists, "margin": float(margin), "confidence": float(confidence)}
        return best, float(confidence), detail

    def is_confident(self, descriptor):
        _, conf, _ = self.predict(descriptor)
        return conf >= self.min_confidence

    # ---- persistence ----
    def to_json(self):
        return json.dumps({"X": self.X.tolist(), "y": list(self.y),
                           "min_confidence": self.min_confidence, "match_dims": self.match_dims})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        r = cls(min_confidence=d.get("min_confidence", 0.5), match_dims=d.get("match_dims", 3))
        r.X = np.array(d["X"], float).reshape(-1, 5) if d["X"] else np.zeros((0, 5))
        r.y = list(d["y"])
        r._fit()
        return r


def default_router():
    """A warm ContractRouter for geometric-data contract selection, seeded from REAL cross-domain
    outcomes plus a few controlled archetypes, so learned routing is accurate out of the box.

    The corpus spans two physical domains -- molecular geometry (QM7 real coordinates, three target types:
    radius of gyration -> equivariant, bond count -> graph, mean atomic number -> set; plus full-budget
    bake-off winners on rMD17 and QM7-equiv, both equivariant) and particle point clouds (JetNet real jets:
    jet width -> equivariant, total pt -> set). Matching uses only the dimensionless proxy-correlation
    features (geo/topo/set), which transfer across domains; the raw size stats are stored but not matched
    on, since a molecule's bond count and a cloud's proximity-edge count are on incomparable scales (a
    measured cross-domain transfer failure; see tests/learned_contract_routing.md). Real bake-off outcomes
    accumulate on top via .add()."""
    r = ContractRouter(min_confidence=0.4, match_dims=3)
    # [geo_proxy, topo_proxy, set_proxy, avgN/10, avgE/20]; labels are the contract the bake-off chooses
    seeds = [
        # --- controlled archetypes (clean geo/topo/set signatures) ---
        ([0.95, 0.20, 0.08, 0.9, 0.5], "equivariant"),
        ([0.90, 0.25, 0.10, 0.7, 0.4], "equivariant"),
        ([0.22, 0.97, 0.09, 0.9, 0.6], "graph"),
        ([0.28, 0.92, 0.12, 0.7, 0.5], "graph"),
        ([0.06, 0.10, 0.98, 0.8, 0.3], "set"),
        ([0.10, 0.12, 0.93, 0.6, 0.2], "set"),
        # --- REAL molecular geometry (QM7 coordinates, three target types) ---
        ([1.00, 0.16, 0.27, 1.32, 0.04], "equivariant"),   # QM7 radius of gyration (geometric)
        ([0.18, 0.96, 0.23, 1.32, 0.04], "graph"),         # QM7 bond count (topological)
        ([0.18, 0.19, 1.00, 1.32, 0.04], "set"),           # QM7 mean atomic number (set)
        # --- REAL full-budget bake-off winners on molecular energy/force targets ---
        ([0.15, 0.21, 0.00, 0.90, 1.54], "equivariant"),   # rMD17-ethanol
        ([0.62, 0.72, 0.00, 1.47, 0.95], "equivariant"),   # QM7-equiv
        # --- REAL particle point clouds (JetNet, a DIFFERENT physical domain) ---
        ([0.98, 0.81, 0.28, 2.99, 21.3], "equivariant"),   # jet width (geometric)
        ([0.30, 0.23, 0.96, 2.99, 21.3], "set"),           # total pt (set)
    ]
    for d, c in seeds:
        r.add(d, c)
    return r
