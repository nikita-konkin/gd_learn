"""Совместимость со средой браузера.

`stlite` подсовывает заглушку вместо `pyarrow`: настоящий пакет в сборку не
влезает, а `Streamlit` его импортирует. У заглушки нет классов `Table`,
`RecordBatch`, `Array` и `ChunkedArray`.

`scikit-learn` на них натыкается. В `sklearn/utils/_dataframe.py` есть

    def is_pyarrow_data(X):
        pa = sys.modules["pyarrow"]
        return isinstance(X, (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray))

то есть проверка «а не стрелочные ли это данные» выполняется каждый раз, когда
`sklearn` разбирает входные данные — и на заглушке падает с `AttributeError`
ещё до начала обучения. Ломается любой вызов `sklearn` в браузере, а не только
наш.

Лечение — дописать заглушке недостающие имена. Каждое становится пустым
классом, экземпляров у которого не бывает, поэтому `isinstance` честно
возвращает `False`: стрелочных данных здесь и правда нет.
"""

from __future__ import annotations

import sys

PYARROW_TYPES = ("Table", "RecordBatch", "Array", "ChunkedArray")


def patch_pyarrow_stub() -> list[str]:
    """Дописать заглушке `pyarrow` недостающие типы. Возвращает список добавленных.

    Вне браузера, где `pyarrow` либо настоящий, либо не импортирован вовсе,
    ничего не делает.
    """
    module = sys.modules.get("pyarrow")
    if module is None:
        return []

    added = []
    for name in PYARROW_TYPES:
        if not hasattr(module, name):
            # Пустой класс: ни один объект его экземпляром не является.
            setattr(module, name, type(name, (), {}))
            added.append(name)
    return added
