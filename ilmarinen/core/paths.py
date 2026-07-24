"""Portable filesystem locations for caches, downloads, and optional offline data.

Every dataset loader and runner routes its scratch/cache paths through here so the
package runs unchanged on any machine -- no hard-coded ``/tmp`` or sandbox-specific
directories. Resolution order for the base data directory:

  1. ``$ILMARINEN_DATA_DIR`` if set (explicit override), else
  2. ``<os-temp-dir>/ilmarinen_data`` (e.g. ``/tmp/ilmarinen_data`` on Linux,
     the user's temp dir on macOS/Windows).

The optional *uploads* directory is a place to drop pre-downloaded dataset files
for fully offline use; it defaults to ``<base>/uploads`` and can be overridden with
``$ILMARINEN_UPLOADS_DIR``. Nothing here creates the uploads dir or requires it to
exist -- loaders treat it purely as a fallback source.
"""

from __future__ import annotations

import os
import tempfile


def data_dir() -> str:
    """Base cache directory for anything the loaders download. Created on demand."""
    d = os.environ.get("ILMARINEN_DATA_DIR") or os.path.join(tempfile.gettempdir(), "ilmarinen_data")
    os.makedirs(d, exist_ok=True)
    return d


def uploads_dir() -> str:
    """Optional directory holding pre-fetched dataset files for offline runs.

    Not created automatically (its absence is fine -- it is only ever read from).
    """
    return os.environ.get("ILMARINEN_UPLOADS_DIR") or os.path.join(data_dir(), "uploads")


def cache_path(*parts: str) -> str:
    """Join ``parts`` under the base data directory."""
    return os.path.join(data_dir(), *parts)
