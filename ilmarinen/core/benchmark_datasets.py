"""Loaders for two more standard-suite real datasets (natural sciences / medicine):

- ECG5000  : cardiac ECG beats (medicine), UCR format. Quasi-periodic 140-length series, 5 classes
             (normal + 4 arrhythmia types). For the SEQUENCE schema (tests conv / spectral).
- OrganMNIST3D : volumetric CT scans of 11 body organs (radiology), 28x28x28. For the VOLUMETRIC
                 schema. Standard MedMNIST-3D npz layout.
"""

from __future__ import annotations

import numpy as np


def load_ecg5000(path_dir=None):
    """Load ECG5000 from the UCR .txt files (class label in column 0, then 140 samples).
    Returns (X_train, y_train, X_test, y_test): X (n, 140) float32, y (n,) int64 in [0,4].
    UCR ECG5000 ships 500 train / 4500 test."""

    def _read(fn):
        arr = np.loadtxt(fn).astype(np.float32)
        y = arr[:, 0].astype(np.int64) - 1  # UCR labels are 1-based
        X = arr[:, 1:]
        return X, y

    if path_dir is None:
        from .data_sources import ecg5000_dir

        path_dir = ecg5000_dir()
    Xtr, ytr = _read(f"{path_dir}/ECG5000_TRAIN.txt")
    Xte, yte = _read(f"{path_dir}/ECG5000_TEST.txt")
    return Xtr, ytr, Xte, yte


def load_bloodmnist(path=None):
    """Load BloodMNIST (MedMNIST-2D). Returns dict with train/val/test images (N,28,28,3) uint8->float32
    and labels (N,) int64 in [0,7] (8 blood-cell classes). RGB microscopy of peripheral blood cells --
    a real COLOR medical image dataset for the spatial schema (complements grayscale MNIST). Fetched via
    the medmnist package (auto-download), falling back to an uploaded bloodmnist.npz."""
    if path is not None:
        d = np.load(path)
    else:
        from .data_sources import medmnist_arrays

        d = medmnist_arrays("bloodmnist")
    out = {}
    for split in ("train", "val", "test"):
        # (N,28,28,3) uint8 -> (N,3,28,28) float32, channels-first for conv2d
        out[f"{split}_x"] = np.asarray(d[f"{split}_images"], dtype=np.float32).transpose(0, 3, 1, 2)
        out[f"{split}_y"] = np.asarray(d[f"{split}_labels"]).reshape(-1).astype(np.int64)
    out["n_classes"] = 8
    out["class_names"] = [
        "basophil",
        "eosinophil",
        "erythroblast",
        "immature granulocyte",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    ]
    return out


def load_organmnist3d(path=None):
    """Load OrganMNIST3D (MedMNIST-3D). Returns dict with train/val/test images (N,28,28,28) uint8 and
    labels (N,) int64 in [0,10] (11 organ classes). Images are 3D CT bounding boxes of body organs. Fetched via
    the medmnist package (auto-download), falling back to an uploaded organmnist3d.npz."""
    if path is not None:
        d = np.load(path)
    else:
        from .data_sources import medmnist_arrays

        d = medmnist_arrays("organmnist3d")
    out = {}
    for split in ("train", "val", "test"):
        out[f"{split}_x"] = np.asarray(d[f"{split}_images"], dtype=np.float32)
        out[f"{split}_y"] = np.asarray(d[f"{split}_labels"]).reshape(-1).astype(np.int64)
    out["n_classes"] = 11
    out["class_names"] = [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen",
    ]
    return out
