"""Deprecation helper for the legacy / pre-AllGraph exports (legacy-audit deprecation cycle).

These names still work for one release so downstream code does not break, but they emit a
``DeprecationWarning`` on access and are scheduled for removal. AllGraph is the supported interface;
the legacy subsystem lives under ``ilmarinen.legacy``.
"""

import warnings


def warn_legacy(name, source, stacklevel=3):
    """Emit a DeprecationWarning for legacy export ``name``, pointing at its canonical ``source`` location.

    Called from a module ``__getattr__`` (PEP 562), so the default ``stacklevel=3`` attributes the warning
    to the caller's line (warn_legacy -> __getattr__ -> caller).
    """
    warnings.warn(
        f"'{name}' is a legacy/pre-AllGraph export scheduled for removal in a future release; "
        f"import it from {source} if you still need it (AllGraph is the supported interface).",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
