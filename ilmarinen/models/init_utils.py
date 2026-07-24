"""Criticality-aware linear-layer initialization shared across the schemas.

Extracted from the legacy networks.py (the pre-AllGraph model zoo, now under ilmarinen/legacy) so the
current schemas depend on this small shared util rather than on that legacy module -- keeping AllGraph's
build path free of legacy imports.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _init_linear(lin: nn.Linear, sigma_w2: float, sigma_b2: float = 0.0, scale: float = 1.0):
    """Gaussian init with weight variance sigma_w^2 / fan_in (edge-of-chaos scaling) and optional bias
    variance sigma_b^2 (zeroed when sigma_b2 == 0). `scale` multiplies the weight std (e.g. 0.5 for a
    near-identity residual branch)."""
    with torch.no_grad():
        fan_in = lin.weight.shape[1]
        lin.weight.normal_(0, np.sqrt(sigma_w2 / fan_in) * scale)
        if lin.bias is None:
            return
        if sigma_b2 > 0:
            lin.bias.normal_(0, np.sqrt(sigma_b2))
        else:
            lin.bias.zero_()
