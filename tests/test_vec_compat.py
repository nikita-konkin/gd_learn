"""Compatibility with the ``pyarrow`` stub stlite substitutes.

Without this shim every ``sklearn`` call in the browser fails, not only ours:
``sklearn.utils._dataframe.is_pyarrow_data`` reaches for ``pa.RecordBatch``
while inspecting its input.
"""

import sys
import types

import numpy as np
import pytest

from vec_playground.compat import PYARROW_TYPES, patch_pyarrow_stub


def _is_pyarrow_data():
    """The sklearn function that trips over the stub.

    It lives in a private module and is not in every version — scikit-learn 1.7
    does not have it yet, and there is nothing to check there. Imported lazily
    so the tests collect on any version.
    """
    dataframe = pytest.importorskip(
        "sklearn.utils._dataframe",
        reason="this version of scikit-learn has no pyarrow check yet",
    )
    return dataframe.is_pyarrow_data


@pytest.fixture
def stub_pyarrow(monkeypatch):
    """Replace ``pyarrow`` with a stub lacking the classes, as stlite does."""
    stub = types.ModuleType("pyarrow")
    monkeypatch.setitem(sys.modules, "pyarrow", stub)
    return stub


def test_sklearn_breaks_on_the_bare_stub(stub_pyarrow):
    """Show the breakage first, or the shim would be verifying nothing."""
    is_pyarrow_data = _is_pyarrow_data()

    with pytest.raises(AttributeError, match="|".join(PYARROW_TYPES)):
        is_pyarrow_data(np.zeros(3))


def test_sklearn_breaks_on_the_stlite_stub(stub_pyarrow):
    """stlite has ``Table`` — Streamlit itself uses it — but not the others."""
    is_pyarrow_data = _is_pyarrow_data()
    stub_pyarrow.Table = type("Table", (), {})

    with pytest.raises(AttributeError, match="RecordBatch"):
        is_pyarrow_data(np.zeros(3))


def test_patch_adds_every_missing_type(stub_pyarrow):
    added = patch_pyarrow_stub()

    assert set(added) == set(PYARROW_TYPES)
    assert all(hasattr(stub_pyarrow, name) for name in PYARROW_TYPES)


def test_sklearn_works_after_the_patch(stub_pyarrow):
    is_pyarrow_data = _is_pyarrow_data()
    patch_pyarrow_stub()

    assert is_pyarrow_data(np.zeros(3)) is False
    assert is_pyarrow_data([1, 2, 3]) is False


def test_patch_is_idempotent(stub_pyarrow):
    patch_pyarrow_stub()
    sentinel = stub_pyarrow.RecordBatch

    assert patch_pyarrow_stub() == []
    assert stub_pyarrow.RecordBatch is sentinel, "a second pass must not replace the types"


def test_patch_leaves_a_real_pyarrow_alone(monkeypatch):
    real = types.ModuleType("pyarrow")
    for name in PYARROW_TYPES:
        setattr(real, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "pyarrow", real)

    assert patch_pyarrow_stub() == []


def test_patch_is_a_noop_without_pyarrow(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)

    assert patch_pyarrow_stub() == []
