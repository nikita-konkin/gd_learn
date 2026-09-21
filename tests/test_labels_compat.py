"""The pyarrow stub shim in the data-and-annotation package.

Behaviour is covered in full by ``test_vec_compat``; what matters here is that
this third copy has not drifted from the first. The site builder ships one
package per app, so a fix applied to one copy and not the others would leave
sklearn broken in the browser for part of the site.
"""

import ast
import inspect
import sys
import textwrap
import types

from labels_playground import compat as labels_compat
from vec_playground import compat as vec_compat


def _logic(function) -> str:
    """The function's syntax tree with its docstring dropped: prose may differ, code may not."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if ast.get_docstring(tree) is not None:
        tree.body = tree.body[1:]
    return ast.dump(tree)


def test_the_copy_has_not_drifted():
    assert labels_compat.PYARROW_TYPES == vec_compat.PYARROW_TYPES
    assert _logic(labels_compat.patch_pyarrow_stub) == _logic(vec_compat.patch_pyarrow_stub)


def test_patch_adds_every_missing_type(monkeypatch):
    stub = types.ModuleType("pyarrow")
    monkeypatch.setitem(sys.modules, "pyarrow", stub)

    assert set(labels_compat.patch_pyarrow_stub()) == set(labels_compat.PYARROW_TYPES)
