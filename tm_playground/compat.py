"""Browser-environment shim.

``stlite`` substitutes a stub for ``pyarrow``: the real package does not fit in
the bundle, but ``streamlit`` imports it. The stub has no ``Table``,
``RecordBatch``, ``Array`` or ``ChunkedArray``.

``scikit-learn`` walks straight into that. ``sklearn/utils/_dataframe.py`` has

    def is_pyarrow_data(X):
        pa = sys.modules["pyarrow"]
        return isinstance(X, (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray))

and that check runs every time ``sklearn`` inspects its input, so against the
stub it raises ``AttributeError`` before any fitting begins. It breaks every
``sklearn`` call in the browser, not just ours.

The cure is to give the stub the missing names. Each becomes an empty class
that nothing is ever an instance of, so ``isinstance`` honestly answers False:
there really is no arrow data here.

The site builder ships one package per app, so a module two apps need has to
exist in both: this is ``vec_playground/compat.py`` again. ``test_tm_compat``
compares the source of the two functions and fails if they drift apart.
"""

from __future__ import annotations

import sys

PYARROW_TYPES = ("Table", "RecordBatch", "Array", "ChunkedArray")


def patch_pyarrow_stub() -> list[str]:
    """Add the missing types to the ``pyarrow`` stub. Returns the names added.

    Outside the browser, where ``pyarrow`` is either real or never imported,
    this does nothing.
    """
    module = sys.modules.get("pyarrow")
    if module is None:
        return []

    added = []
    for name in PYARROW_TYPES:
        if not hasattr(module, name):
            # An empty class: no object is an instance of it.
            setattr(module, name, type(name, (), {}))
            added.append(name)
    return added
