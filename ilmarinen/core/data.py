"""Data loading and preprocessing for the meta-optimizer framework.

Fetches Fashion-MNIST from the Zalando GitHub mirror (works behind restricted
networks where OpenML is blocked) and caches locally. Provides standardized
tensors plus convenient subset / binary-task helpers used by the validation
pipelines.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

_MIRROR = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for _, fname in _FILES.items():
        dst = os.path.join(cache_dir, fname)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(_MIRROR + fname, dst)


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * r * c)
    return np.frombuffer(buf, np.uint8).reshape(n, r * c).astype(np.float32)


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
    return np.frombuffer(buf, np.uint8).astype(np.int64)


class FashionMNIST:
    """Standardized Fashion-MNIST with per-pixel zero-mean/unit-variance scaling.

    Attributes
    ----------
    Xtr, ytr, Xte, yte : np.ndarray
        Train/test images (float32, standardized) and integer labels.
    """

    n_features = 784
    n_classes = 10

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            from .paths import cache_path
            cache_dir = cache_path("fmnist")
        _download(cache_dir)
        Xtr = _read_images(os.path.join(cache_dir, _FILES["train_images"]))
        ytr = _read_labels(os.path.join(cache_dir, _FILES["train_labels"]))
        Xte = _read_images(os.path.join(cache_dir, _FILES["test_images"]))
        yte = _read_labels(os.path.join(cache_dir, _FILES["test_labels"]))
        self._mu = Xtr.mean(0, keepdims=True)
        self._sd = Xtr.std(0, keepdims=True) + 1e-6
        self.Xtr = (Xtr - self._mu) / self._sd
        self.Xte = (Xte - self._mu) / self._sd
        self.ytr, self.yte = ytr, yte

    def balanced_subset(self, per_class: int = 800, split: str = "train"):
        """Return (X, y) with `per_class` examples of each of the 10 classes."""
        X, y = (self.Xtr, self.ytr) if split == "train" else (self.Xte, self.yte)
        idx = np.concatenate([np.where(y == k)[0][:per_class] for k in range(self.n_classes)])
        return X[idx], y[idx]

    def binary_task(self, cls_a: int, cls_b: int, per_class: int = 1500):
        """Return (Xtr, ytr, Xte, yte) for a +/-1 binary problem between two classes.

        Used by the width-sparsity certificate, which is exact for the
        two-layer / scalar-output setting.
        """
        ia = np.where(self.ytr == cls_a)[0][:per_class]
        ib = np.where(self.ytr == cls_b)[0][:per_class]
        Xtr = np.vstack([self.Xtr[ia], self.Xtr[ib]])
        ytr = np.concatenate([np.ones(len(ia)), -np.ones(len(ib))])
        iate = np.where(self.yte == cls_a)[0]
        ibte = np.where(self.yte == cls_b)[0]
        Xte = np.vstack([self.Xte[iate], self.Xte[ibte]])
        yte = np.concatenate([np.ones(len(iate)), -np.ones(len(ibte))])
        return Xtr, ytr, Xte, yte


    def sequential_subset(self, per_class: int = 500, split: str = "train"):
        """Return (X_seq, y) with X_seq shaped (n, 784, 1): pixels as a timestep
        stream for sequential (recurrent) tasks. Same pixels as balanced_subset,
        just reshaped so each of the 784 pixels is one timestep of a length-784
        sequence with input dim 1.
        """
        X, y = self.balanced_subset(per_class=per_class, split=split)
        return X.reshape(X.shape[0], 784, 1), y
