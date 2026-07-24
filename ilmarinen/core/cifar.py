"""CIFAR-10 data loading for the convolution primitive test.

Loads from the extracted GitHub image mirror (32x32 JPGs in class folders),
standardizes per-channel, and returns tensors shaped for spatial models:
(n, 3, 32, 32). Matches the FashionMNIST interface style (balanced_subset).

Because loading all 60k JPGs is slow, this loads a requested subset directly
from disk rather than caching the full set -- the schema test needs only
a few thousand images.
"""
from __future__ import annotations
import os
import numpy as np

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def _load_image(path):
    """Load a 32x32 RGB JPG to a (3, 32, 32) float array via skimage."""
    from skimage.io import imread
    img = imread(path)                       # (32, 32, 3) uint8
    if img.ndim == 2:                        # grayscale safety
        img = np.stack([img] * 3, axis=-1)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


class CIFAR10:
    """CIFAR-10 from the extracted image directory.

    Parameters
    ----------
    root : path to the extracted 'CIFAR-10-images-master' directory.
    """

    n_classes = 10
    img_shape = (3, 32, 32)

    def __init__(self, root: str | None = None):
        if root is None:
            from .paths import cache_path
            root = cache_path("CIFAR-10-images-master")
        self.root = root
        if not os.path.isdir(os.path.join(root, "train")):
            raise FileNotFoundError(f"CIFAR-10 train dir not found under {root}")
        # per-channel normalization stats (standard CIFAR-10 values)
        self._mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(3, 1, 1)
        self._std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(3, 1, 1)

    def balanced_subset(self, per_class: int = 300, split: str = "train"):
        """Return (X, y): X shaped (n, 3, 32, 32) standardized, y integer labels."""
        base = os.path.join(self.root, split)
        Xs, ys = [], []
        for ci, cname in enumerate(CIFAR_CLASSES):
            cdir = os.path.join(base, cname)
            files = sorted(os.listdir(cdir))[:per_class]
            for f in files:
                img = _load_image(os.path.join(cdir, f))
                Xs.append((img - self._mean) / self._std)
                ys.append(ci)
        X = np.stack(Xs).astype(np.float32)
        y = np.array(ys, np.int64)
        # shuffle so classes aren't blocked
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def flat_subset(self, per_class: int = 300, split: str = "train"):
        """Flattened (n, 3072) version for dense-only baselines."""
        X, y = self.balanced_subset(per_class, split)
        return X.reshape(X.shape[0], -1), y
