"""JetNet loader -- high-energy-physics jet tagging as an UNORDERED SET task (for the set schema).

Each jet is a variable-size SET of particles (up to 30), each particle a vector [eta_rel, phi_rel,
pt_rel]; a 4th column is a presence MASK (jets are zero-padded to 30 particles). The task is to classify
the jet's origin: g (gluon), q (light quark), t (top quark), w (W boson), z (Z boson). This is the
canonical particle-physics unordered-set benchmark (DeepSets / ParticleNet domain): the physics is
permutation-invariant in the particle index, exactly the set contract's symmetry.

Files: g.hdf5, q.hdf5, t.hdf5, w.hdf5, z.hdf5, each with:
  particle_features (N, 30, 4)  -- [eta_rel, phi_rel, pt_rel, mask]
  jet_features      (N, 4)      -- [pt, eta, mass, n_particles]  (not used for the set task)
"""
from __future__ import annotations

import numpy as np

JET_CLASSES = ("g", "q", "t", "w", "z")
_CLASS_NAMES = {"g": "gluon", "q": "light quark", "t": "top", "w": "W boson", "z": "Z boson"}


def load_jetnet(upload_dir=None, classes=JET_CLASSES, n_per_class=None,
                min_pt=0.0):
    """Load JetNet jets as sets. Returns (particles, mask, labels, class_names):
      particles : (n_jets, 30, 3) float32   -- [eta_rel, phi_rel, pt_rel] per particle
      mask      : (n_jets, 30)   bool        -- True where a real particle is present
      labels    : (n_jets,)      int64       -- index into `classes`
      class_names: list[str]                 -- human-readable, aligned with label index
    n_per_class caps jets per class (balanced subsample) for tractable runs.

    The .hdf5 files are resolved via data_sources.jetnet_hdf5_dir, which auto-downloads them from the
    JetNet Zenodo record and caches them; an uploaded copy in $ILMARINEN_UPLOADS_DIR (g/q/t/w/z.hdf5) is used
    first for offline runs. Pass `upload_dir` to read from an explicit directory instead."""
    import h5py
    if upload_dir is None:
        from .data_sources import jetnet_hdf5_dir
        upload_dir = jetnet_hdf5_dir(classes=tuple(classes))
    parts, masks, labels = [], [], []
    for ci, c in enumerate(classes):
        with h5py.File(f"{upload_dir}/{c}.hdf5", "r") as f:
            pf = f["particle_features"][:]                       # (N,30,4)
        if n_per_class is not None and len(pf) > n_per_class:
            idx = np.random.RandomState(0).permutation(len(pf))[:n_per_class]
            pf = pf[idx]
        feats = pf[:, :, :3].astype(np.float32)                  # [eta,phi,pt]
        m = pf[:, :, 3].astype(bool)                             # presence mask
        parts.append(feats); masks.append(m)
        labels.append(np.full(len(pf), ci, dtype=np.int64))
    particles = np.concatenate(parts, 0)
    mask = np.concatenate(masks, 0)
    labels = np.concatenate(labels, 0)
    names = [_CLASS_NAMES[c] for c in classes]
    return particles, mask, labels, names


def jetnet_to_set_batch(particles, mask, indices):
    """Flatten a batch of jets into the set-schema contract (X (N,F), batch (N,), n_sets).
    Drops masked (padding) particles so each set has only its real particles -- the set schema is
    size-agnostic. Returns (X, batch, n_sets) as torch tensors."""
    import torch
    Xs, batch = [], []
    for j, i in enumerate(indices):
        m = mask[i]
        pts = particles[i][m]                                   # (n_real, 3)
        Xs.append(pts)
        batch.append(np.full(len(pts), j, dtype=np.int64))
    X = torch.tensor(np.concatenate(Xs, 0))
    b = torch.tensor(np.concatenate(batch, 0))
    return X, b, len(indices)
