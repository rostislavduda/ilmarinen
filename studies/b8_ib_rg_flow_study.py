"""B8: IB-as-RG flow -- making "redundancy reduction as RG" quantitative.

A STUDY + a small new diagnostic module (core/ib_rg_flow.py). Demonstrates the exact Gaussian-IB <-> RG
correspondence (Chechik et al. JMLR 2005; Kline & Palmer, arXiv:2107.13700): the effective dimension flows
along a coarse-graining scale beta as a staircase with transitions at beta_c = 1/(1-lambda), and the same
Gaussian-IB spectrum, applied across a trained network's layers, measures the RG coarse-graining of depth
toward the target. See tests/b8_ib_rg_flow.md for the full write-up and honest scope.

Run: python studies/b8_ib_rg_flow_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys

import numpy as np

sys.path.insert(0, ".")
from ilmarinen.core.ib_rg_flow import critical_betas, gib_spectrum, ib_rg_flow, layer_rg_flow  # noqa: E402


def controlled_gaussian(rho=(0.95, 0.8, 0.5, 0.2, 0.05), n=4000, seed=0):
    """Verify the GIB closed form and the beta staircase on data with KNOWN canonical correlations, and check
    the semigroup (nesting) property that makes it a genuine RG flow."""
    rng = np.random.RandomState(seed)
    rho = np.asarray(rho); d = len(rho)
    Z = rng.randn(n, d); Y = Z.copy()
    X = rho[None, :] * Y + np.sqrt(1 - rho ** 2)[None, :] * rng.randn(n, d)
    Q, _ = np.linalg.qr(rng.randn(d, d)); X = X @ Q.T
    flow = ib_rg_flow(X, Y)
    print("Controlled Gaussian (known canonical correlations):")
    print(f"  true rho                 : {np.round(rho, 3)}")
    print(f"  recovered canonical corr : {np.round(flow['canonical_corr'], 3)}")
    print(f"  critical betas (transitions): {np.round(flow['critical_betas'], 2)}")
    print(f"  d_IB flow range          : {flow['d_IB'].min()}..{flow['d_IB'].max()}")
    lam, _ = gib_spectrum(X, Y); bc = critical_betas(lam)
    sets = [set(np.where(b > bc)[0]) for b in (2.0, 5.0, 30.0)]
    nested = all(sets[i].issubset(sets[i + 1]) for i in range(len(sets) - 1))
    print(f"  semigroup (nested retained-mode sets across beta)? {nested}")


def layer_flow_demo(n=2000, d=12, seed=0):
    """Show the GIB flow across a trained MLP's layers on a rank-3 target: the target becomes progressively
    more linearly decodable (top canonical corr -> 1), the measurable RG coarse-graining of depth."""
    import torch
    import torch.nn as nn
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.tanh(X[:, 0] + X[:, 1]) + 0.5 * np.tanh(X[:, 2] - X[:, 3]) + 0.3 * X[:, 4]).astype(np.float32)
    Xt = torch.tensor(X); yt = torch.tensor(y).unsqueeze(1)

    class Net(nn.Module):
        def __init__(self, d, h=24, L=4):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d if i == 0 else h, h) for i in range(L)])
            self.out = nn.Linear(h, 1)

        def forward(self, x, rh=False):
            hs = []
            for lin in self.layers:
                x = torch.tanh(lin(x)); hs.append(x)
            o = self.out(x)
            return (o, hs) if rh else o

    torch.manual_seed(seed)
    net = Net(d, 24, 4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    for _ in range(400):
        opt.zero_grad(); ((net(Xt) - yt) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        _, hs = net(Xt, rh=True)
    lf = layer_rg_flow([h.numpy() for h in hs], y)
    print("\nRG-flow across a trained net's layers (coarse-graining toward the target):")
    for r in lf["layers"]:
        print(f"  layer {r['layer'] + 1}: top canonical corr={r['top_canonical_corr']:.3f}, "
              f"relevant info={r['relevant_information']:.3f}, #informative={r['n_informative']}")
    print(f"  monotone rise in decodability? {lf['top_canonical_corr_monotone']}")


if __name__ == "__main__":
    controlled_gaussian()
    layer_flow_demo()
    print("\n(see tests/b8_ib_rg_flow.md for the real-data ECG5000 cascade and honest scope)")
