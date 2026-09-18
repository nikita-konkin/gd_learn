"""Совместимость с заглушкой `pyarrow`, которую подсовывает stlite.

Без этой заплатки в браузере падает любой вызов `sklearn`, а не только наш:
`sklearn.utils._dataframe.is_pyarrow_data` обращается к `pa.RecordBatch` при
разборе входных данных.
"""

import sys
import types

import numpy as np
import pytest
from sklearn.utils._dataframe import is_pyarrow_data

from vec_playground.compat import PYARROW_TYPES, patch_pyarrow_stub


@pytest.fixture
def stub_pyarrow(monkeypatch):
    """Подменить `pyarrow` заглушкой без нужных классов — как в stlite."""
    stub = types.ModuleType("pyarrow")
    monkeypatch.setitem(sys.modules, "pyarrow", stub)
    return stub


def test_sklearn_breaks_on_the_bare_stub(stub_pyarrow):
    """Сначала показываем саму поломку, иначе заплатка проверяет пустоту."""
    with pytest.raises(AttributeError, match="|".join(PYARROW_TYPES)):
        is_pyarrow_data(np.zeros(3))


def test_sklearn_breaks_on_the_stlite_stub(stub_pyarrow):
    """У stlite `Table` есть — его использует сам Streamlit, — а остальных нет."""
    stub_pyarrow.Table = type("Table", (), {})

    with pytest.raises(AttributeError, match="RecordBatch"):
        is_pyarrow_data(np.zeros(3))


def test_patch_adds_every_missing_type(stub_pyarrow):
    added = patch_pyarrow_stub()

    assert set(added) == set(PYARROW_TYPES)
    assert all(hasattr(stub_pyarrow, name) for name in PYARROW_TYPES)


def test_sklearn_works_after_the_patch(stub_pyarrow):
    patch_pyarrow_stub()

    assert is_pyarrow_data(np.zeros(3)) is False
    assert is_pyarrow_data([1, 2, 3]) is False


def test_patch_is_idempotent(stub_pyarrow):
    patch_pyarrow_stub()
    sentinel = stub_pyarrow.RecordBatch

    assert patch_pyarrow_stub() == []
    assert stub_pyarrow.RecordBatch is sentinel, "второй проход не должен подменять типы"


def test_patch_leaves_a_real_pyarrow_alone(monkeypatch):
    real = types.ModuleType("pyarrow")
    for name in PYARROW_TYPES:
        setattr(real, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "pyarrow", real)

    assert patch_pyarrow_stub() == []


def test_patch_is_a_noop_without_pyarrow(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)

    assert patch_pyarrow_stub() == []
