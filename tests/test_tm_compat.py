"""The pyarrow stub shim, and the fact that it stays the same in both packages.

The behaviour itself is covered in full by ``test_vec_compat``. What is worth
checking here is that the two copies have not drifted: the site builder ships
one package per app, so this module exists twice, and a fix applied to one copy
and not the other would leave sklearn broken in the browser for half the site.
"""

import ast
import inspect
import sys
import textwrap
import types

import numpy as np

from tm_playground import compat as tm_compat
from vec_playground import compat as vec_compat


def _logic(function) -> str:
    """The function's syntax tree with its docstring dropped.

    Comparing source text would fail on the prose alone — the two copies are
    documented in different languages — and prose drifting apart harms nobody.
    What must not drift is the code.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if ast.get_docstring(tree) is not None:
        tree.body = tree.body[1:]
    return ast.dump(tree)


def test_the_two_copies_have_not_drifted():
    assert tm_compat.PYARROW_TYPES == vec_compat.PYARROW_TYPES
    assert _logic(tm_compat.patch_pyarrow_stub) == _logic(
        vec_compat.patch_pyarrow_stub
    ), "vec_playground/compat.py and tm_playground/compat.py must patch identically"


def test_patch_adds_every_missing_type(monkeypatch):
    stub = types.ModuleType("pyarrow")
    monkeypatch.setitem(sys.modules, "pyarrow", stub)

    added = tm_compat.patch_pyarrow_stub()

    assert set(added) == set(tm_compat.PYARROW_TYPES)
    assert not isinstance(np.zeros(3), stub.RecordBatch)


def test_patch_is_a_noop_without_pyarrow(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)

    assert tm_compat.patch_pyarrow_stub() == []
