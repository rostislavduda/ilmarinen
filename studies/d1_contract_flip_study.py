"""D1 near-tied contract-flip study: is the functional code length ever decision-relevant?

A STUDY (not a package feature). D1 fused the singular (functional) code length omega_func =
max(lambda,0)*log n into the priced contract objective J = R + mu_c*Omega, where Omega gains a term
(s_mu/mu_c)*omega_func for each converged contract. On QM7-equiv, D1 fired and its non-negativity
guard worked, but it never FLIPPED the contract winner: the structural spread there (set 0 / graph 32 /
equivariant 127 nats) dwarfs omega_func (<= ~4 nats), so the functional term is second-order.

This study asks the sharper question the QM7 run left open: is there ANY converged, near-tied regime
where the functional term flips a contract decision? The arithmetic says a flip needs two contracts
near-tied in BOTH structural Omega AND fit R, converging to solutions of DIFFERENT singularity.

Construction of the structural near-tie: a COMPLETE graph has Omega_struct(graph) = E*log(P/E) -> 0 as
E -> P = N(N-1)/2 (log 1 = 0), matching Omega_struct(set) = 0. So on fully-connected relational data,
set and graph are structurally tied at ~0 nats; only fit and functional complexity distinguish them.

Findings (see tests/d1_contract_flip.md for the full write-up):
  * The MECHANISM can flip -- but only in the ANTI-CORRELATED corner where the struct-only winner is
    the HIGHER-lambda (more singular) contract, so the functional penalty pushes AGAINST it.
  * Natural training does NOT produce that corner. At a structure+fit tie, the simpler function class
    (the set encoder) both wins the structural tie-break (Omega=0) AND has the LOWER functional
    complexity (lower lambda). The two Occam pressures are ALIGNED, so the functional term can only
    reinforce, never overturn, the structural verdict -- in exactly the regime where it had most
    leverage. Across seeds the winner is `set` at every mu_c, struct-only and struct+func alike.

Conclusion: omega_func is a genuine, correctly-guarded complexity term, but it is decision-relevant
only when fit-quality and functional-simplicity are ANTI-correlated across the tied contracts -- a
corner natural training does not reach here. This bounds D1's role: it refines J and prices effective
DoF honestly, but does not, on realistic converged data, change which contract is chosen.

Run: PYTHONPATH=/tmp/ilmarinen python studies/d1_contract_flip_study.py
"""

import numpy as np
import torch
import torch.nn as nn

from ilmarinen.machinery.contract_mdl import omega_struct, select_contract_mdl
from ilmarinen.machinery.singular_mdl import omega_func, singular_complexity_of


# ------------------------------------------------------------------ data + encoders
def make_data(n_graphs=300, N=6, seed=0):
    """Complete-graph data; permutation-invariant first-moment target expressible by BOTH a mean-pool
    set encoder and a complete-graph encoder, so fits converge to a tight tie."""
    rng = np.random.RandomState(seed)
    P = N * (N - 1) // 2
    feats = rng.randn(n_graphs, N, 3).astype(np.float32)
    m = feats.mean(axis=1)
    y = (np.tanh(m[:, 0]) + 0.5 * m[:, 1] ** 2 - 0.3 * m[:, 2]).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return feats, y, N, P


class SetEncoder(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh())
        self.rho = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))

    def forward(self, x):
        return self.rho(self.phi(x).mean(dim=1)).squeeze(-1)


class GraphEncoder(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(6, w), nn.Tanh())
        self.upd = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh())
        self.rho = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))

    def forward(self, x):
        B, N, _ = x.shape
        xi = x.unsqueeze(2).expand(B, N, N, 3)
        xj = x.unsqueeze(1).expand(B, N, N, 3)
        m = self.msg(torch.cat([xi, xj], dim=-1)).mean(dim=2)
        return self.rho(self.upd(m).mean(dim=1)).squeeze(-1)


def train(model, X, y, epochs=400, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(model(Xt), yt).backward()
        opt.step()
    with torch.no_grad():
        ss_res = float(((model(Xt) - yt) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1 - ss_res / (ss_tot + 1e-12)


def llc_closure(model, X, y):
    Xt, yt = torch.tensor(X), torch.tensor(y)
    lossf = nn.MSELoss()
    return lambda: lossf(model(Xt), yt)


# ------------------------------------------------------------------ the study
def natural_corner(n_seeds=5):
    print("=== NATURAL corner: train set vs graph on a structure+fit tie, measure real lambda ===")
    any_flip = False
    for seed in range(n_seeds):
        X, y, N, P = make_data(n_graphs=300, N=6, seed=seed)
        n = len(X)
        set_net, graph_net = SetEncoder(32), GraphEncoder(32)
        sr = train(set_net, X, y, epochs=400, seed=seed + 10)
        gr = train(graph_net, X, y, epochs=400, seed=seed + 10)
        sc_s = singular_complexity_of(
            set_net, llc_closure(set_net, X, y), n, chains=3, steps=250, burn=80, eps=2e-5, seed=seed
        )
        sc_g = singular_complexity_of(
            graph_net, llc_closure(graph_net, X, y), n, chains=3, steps=250, burn=80, eps=2e-5, seed=seed
        )
        scores = {"set": sr, "graph": gr}
        om_struct = {"set": omega_struct("set", N, P), "graph": omega_struct("graph", N, P)}
        flips = []
        for mc in [0.05, 0.1, 0.2, 0.4]:
            w_s, _ = select_contract_mdl(scores, om_struct, mu_c=mc)
            om_aug = dict(om_struct)
            for c, sc in [("set", sc_s), ("graph", sc_g)]:
                if sc["valid"]:
                    om_aug[c] = om_struct[c] + sc["omega_func"]
            w_f, _ = select_contract_mdl(scores, om_aug, mu_c=mc)
            flips.append(w_f != w_s)
            any_flip = any_flip or (w_f != w_s)
        print(
            f"  seed {seed}: R2 set={sr:.4f} graph={gr:.4f} | lambda set={sc_s['lambda']:+.2f} "
            f"graph={sc_g['lambda']:+.2f} | omega_func set={sc_s['omega_func']:.1f} graph={sc_g['omega_func']:.1f} "
            f"| flips@mu_c[.05,.1,.2,.4]={flips}"
        )
    print(f"  => any flip in natural runs: {any_flip}\n")
    return any_flip


def adversarial_corner():
    print("=== ADVERSARIAL corner: place lambda so the fit-leader is the MORE singular contract ===")
    n = 400
    scores = {"set": 0.9969, "graph": 0.9968}  # set marginally leads fit
    om_struct = {"set": 0.0, "graph": 0.015}  # complete-graph tie
    # fit-leader (set) made deliberately MORE singular than graph:
    lam = {"set": 8.0, "graph": 1.0}
    for mc in [0.05, 0.2]:
        w_s, _ = select_contract_mdl(scores, om_struct, mu_c=mc)
        om_aug = {c: om_struct[c] + omega_func(lam[c], n) for c in scores}
        w_f, df = select_contract_mdl(scores, om_aug, mu_c=mc)
        flip = "  <== FLIP" if w_f != w_s else ""
        print(f"  mu_c={mc}: lambda set={lam['set']} graph={lam['graph']} | struct-only={w_s} +func={w_f}{flip}")
    # threshold: how singular must the fit-leader be to flip at mu_c=0.05?
    mc = 0.05
    thr = None
    for lam_set in np.arange(1.0, 6.01, 0.25):
        om_aug = {"set": 0.0 + omega_func(lam_set, n), "graph": 0.015 + omega_func(1.0, n)}
        w_f, _ = select_contract_mdl(scores, om_aug, mu_c=mc)
        if w_f == "graph":
            thr = lam_set
            break
    print(
        f"  flip threshold at mu_c=0.05: fit-leader lambda >= {thr} (graph lambda=1.0), "
        f"i.e. an omega_func gap of ~{omega_func(thr, n) - omega_func(1.0, n):.1f} nats\n"
    )


def main():
    natural = natural_corner()
    adversarial_corner()
    print(
        "CONCLUSION: the functional term omega_func flips a contract decision only in the "
        "anti-correlated corner (fit-leader = more singular). Natural training aligns the two Occam "
        "pressures (simpler class wins structure AND has lower lambda), so no flip occurs here: "
        f"natural-run flip = {natural}."
    )


if __name__ == "__main__":
    main()
