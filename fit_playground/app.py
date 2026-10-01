"""Streamlit interface for the model-complexity playground.

Three tabs, three handles on the same question — how flexible should a model
be — each opening on the setting its lecture prints. The tabs share no
control: the degree moves only the polynomial, the neighbours and the depth only
their own panel, the penalty only the regression.
"""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

from playground_common.compat import patch_pyarrow_stub

# Before the first sklearn import: stlite's pyarrow stub is missing types that
# sklearn's input check reaches for on every call.
patch_pyarrow_stub()

from fit_playground.models import (  # noqa: E402  — must follow the stub patch
    ALPHAS,
    LECTURE_DEGREES,
    LECTURE_DEPTH,
    MAX_DEGREE,
    Classifier,
    Penalty,
    PolynomialFit,
    feature_names,
    fit_polynomial,
    fresh_data,
    mean_prediction_error,
    moons,
    neighbours,
    penalty,
    polynomial_data,
    tree,
)
from fit_playground.plotting import (  # noqa: E402
    boundary_figure,
    degree_curve_figure,
    paths_figure,
    polynomial_figure,
)
from playground_common.links import other_playgrounds  # noqa: E402
from playground_common.wording import as_printed, plural  # noqa: E402

TITLE = "Сложность модели"

DEGREE_LABEL = "Степень многочлена"
NEIGHBOURS_LABEL = "Соседей, k"
DEPTH_LABEL = "Глубина дерева"
ALPHA_LABEL = "Сила штрафа alpha"

UNLIMITED = "без ограничения"
DEPTHS = ["1", "2", "3", "4", "5", "6", "8", "10", UNLIMITED]
LECTURE_DEGREE = 17
LECTURE_NEIGHBOURS = 1
LECTURE_ALPHA = ALPHAS[-1]
# Lecture 5 also prints the grid point nearest 10, where Lasso selects rather than collapses.
LECTURE_MIDDLE_ALPHA = min(ALPHAS, key=lambda value: abs(value - 10))


@lru_cache(maxsize=MAX_DEGREE + 1)
def _polynomial(degree: int) -> PolynomialFit:
    return fit_polynomial(degree)


@lru_cache(maxsize=128)
def _neighbours(count: int) -> Classifier:
    return neighbours(count)


@lru_cache(maxsize=16)
def _tree(depth: int | None) -> Classifier:
    return tree(depth)


@lru_cache(maxsize=len(ALPHAS))
def _penalty(alpha: float) -> Penalty:
    return penalty(alpha)


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.3g}"


def _controls() -> tuple[int, int, int | None, float]:
    """The sidebar: one control per tab, and each reaches only its own tab."""
    st.sidebar.header("Многочлен")
    degree = st.sidebar.slider(DEGREE_LABEL, 1, MAX_DEGREE, LECTURE_DEGREE)

    st.sidebar.header("Соседи и дерево")
    count = st.sidebar.slider(NEIGHBOURS_LABEL, 1, 99, LECTURE_NEIGHBOURS)
    depth_text = st.sidebar.select_slider(DEPTH_LABEL, DEPTHS, value=str(LECTURE_DEPTH))

    st.sidebar.header("Штраф")
    labels = [_alpha_label(alpha) for alpha in ALPHAS]
    chosen = st.sidebar.select_slider(ALPHA_LABEL, labels, value=_alpha_label(LECTURE_ALPHA))

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/fit"))

    depth = None if depth_text == UNLIMITED else int(depth_text)
    return degree, count, depth, ALPHAS[labels.index(chosen)]


def _error(value: float) -> str:
    """Three decimals for errors on the scale of the data, one for errors far beyond it.

    A degree-17 fit is badly conditioned: the browser's NumPy and a desktop one
    agree on its training error to the digit but part ways in the fourth figure
    of its error on new points (13.156 against 13.155). One decimal is all such
    a number can honestly carry.
    """
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def _polynomial_tab(degree: int) -> None:
    st.markdown("Лекция 4: двадцать пять точек синусоиды с шумом и многочлены разной степени.")
    fit = _polynomial(degree)
    baseline = mean_prediction_error()

    left, middle, right = st.columns(3)
    left.metric("Ошибка на обучении", f"{fit.train_error:.3f}")
    middle.metric("Ошибка на новых точках", _error(fit.fresh_error))
    right.metric("Ответ «среднее» на новых точках", f"{baseline:.3f}")

    train_x, train_y = polynomial_data()
    fresh_x, fresh_y = fresh_data()
    st.plotly_chart(polynomial_figure(fit, train_x, train_y, fresh_x, fresh_y), use_container_width=True)

    if degree in LECTURE_DEGREES:
        st.info(
            f"Степень {degree} есть в лекции 4. Над её графиком напечатано «ошибка на обучении "
            f"{fit.train_error:.3f}» — ровно то, что выше."
        )
    if fit.fresh_error > baseline:
        times = fit.fresh_error / baseline
        st.warning(
            f"Ошибка на обучении {fit.train_error:.3f} — а на новых точках {_error(fit.fresh_error)}, "
            f"в {times:.0f} {plural(round(times), 'раз', 'раза', 'раз')} хуже, чем если бы модель всегда отвечала "
            "средним. Лекция утверждает, что так и будет; здесь это измерено."
        )

    degrees = list(range(1, MAX_DEGREE + 1))
    st.plotly_chart(
        degree_curve_figure(
            degrees,
            [_polynomial(value).train_error for value in degrees],
            [_polynomial(value).fresh_error for value in degrees],
            baseline,
            degree,
        ),
        use_container_width=True,
    )
    st.caption(
        "Ошибка на обучении падает с каждой степенью — и ничего не доказывает. Ошибка на новых точках "
        "сначала падает вместе с ней, потом разворачивается: многочлен начинает повторять шум."
    )


def _classifier_column(model: Classifier, title: str) -> None:
    features, target = moons()
    st.plotly_chart(boundary_figure(model, features, target, title), use_container_width=True)
    left, right = st.columns(2)
    left.metric(f"{title}: на обучении", as_printed(model.train_accuracy))
    right.metric(f"{title}: на проверке", as_printed(model.cv_accuracy))


def _neighbours_tab(count: int, depth: int | None) -> None:
    st.markdown(
        "Лекция 5: метод ближайших соседей и дерево решений на трёхстах точках двух полумесяцев. "
        "«На проверке» — средняя доля правильных по пяти блокам кросс-валидации."
    )
    knn = _neighbours(count)
    model = _tree(depth)
    left, right = st.columns(2)
    with left:
        _classifier_column(knn, f"k = {count}")
    with right:
        _classifier_column(model, f"глубина {depth if depth is not None else '∞'}")

    best = _neighbours(15)
    if count == 1:
        st.caption(
            f"При одном соседе модель отвечает на обучении безошибочно ({as_printed(knn.train_accuracy)}): "
            "каждая точка — сама себе сосед. Лекция называет это переобучением, и граница действительно "
            f"огибает каждый выброс. Но на проверке k = 1 даёт {as_printed(knn.cv_accuracy)} против "
            f"{as_printed(best.cv_accuracy)} у k = 15 — цена переобучения здесь мала. "
            f"Сдвиньте k к 99: недообучение обходится в {as_printed(best.cv_accuracy - _neighbours(99).cv_accuracy)}."
        )
    st.caption(
        f"Дерево глубины {depth if depth is not None else 'без ограничения'}: "
        f"на обучении {as_printed(model.train_accuracy)}, на проверке {as_printed(model.cv_accuracy)}. "
        "Разница между этими числами — то, что обучающая выборка от вас скрывает."
    )


def _penalty_tab(alpha: float) -> None:
    st.markdown(
        "Лекция 5: Ridge и Lasso на таблице diabetes, десять признаков, признаки стандартизованы. "
        "Метрика — R² на отложенных 30 %."
    )
    chosen = _penalty(alpha)
    ridge_paths = [_penalty(value).ridge_weights for value in ALPHAS]
    lasso_paths = [_penalty(value).lasso_weights for value in ALPHAS]
    names = feature_names()

    left, middle, right = st.columns(3)
    left.metric("Lasso обнулил признаков", f"{chosen.lasso_zeroed} из {len(names)}")
    middle.metric("R² Ridge", f"{chosen.ridge_r2:.3f}")
    right.metric("R² Lasso", f"{chosen.lasso_r2:.3f}")

    st.plotly_chart(paths_figure(ALPHAS, ridge_paths, lasso_paths, names, alpha), use_container_width=True)

    if alpha == LECTURE_ALPHA:
        st.info(
            f"Наибольший штраф лекции 5. Лекция печатает «При максимальном штрафе Lasso обнулил "
            f"{chosen.lasso_zeroed} признаков из {len(names)}» — ровно то, что выше."
        )
    elif alpha == LECTURE_MIDDLE_ALPHA:
        st.info(
            f"Этот штраф тоже есть в лекции 5: «При alpha = {alpha:.1f} Lasso обнулил {chosen.lasso_zeroed} "
            f"признаков из {len(names)}» — ровно то, что выше. Это и есть отбор признаков."
        )
    if chosen.lasso_zeroed == len(names):
        first = next(value for value in ALPHAS if _penalty(value).lasso_zeroed == len(names))
        st.warning(
            f"Обнулены все {len(names)} признаков, и R² Lasso — {chosen.lasso_r2:.3f}: модель отвечает одним "
            "числом для всех, это уже не отбор признаков, а отказ от модели. В сетке лекции так происходит "
            f"начиная с alpha ≈ {_alpha_label(first)}. Отбор, ради которого Lasso применяют, виден левее: "
            "сдвиньте штраф к 10."
        )
    st.caption(
        "Одна и та же alpha для Ridge и Lasso означает разную силу: Lasso в scikit-learn делит ошибку на "
        "удвоенный размер выборки, Ridge — нет. Поэтому на общей шкале Lasso успевает обнулить всё, пока Ridge "
        "лишь вдвое сжимает веса. Сравнивать их по одной alpha — значит сравнивать шкалы, а не методы."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Слишком простая модель недообучается, слишком гибкая — запоминает шум. Здесь гибкость задаётся "
        "ручкой, а цена ошибки измеряется на данных, которых модель не видела."
    )

    degree, count, depth, alpha = _controls()
    polynomial, classifiers, penalties = st.tabs(["Многочлен", "Соседи и дерево", "Штраф"])
    with polynomial:
        _polynomial_tab(degree)
    with classifiers:
        _neighbours_tab(count, depth)
    with penalties:
        _penalty_tab(alpha)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не подбирает гиперпараметры автоматически: подбор по сетке с вложенной проверкой — "
            "задание лабораторной работы модуля 2.\n"
            "- Не меняет данные: точки лекций 4 и 5 одни и те же при любых настройках. Новые точки "
            "многочлена взяты из того же закона отдельным генератором.\n"
            "- Не сравнивает ансамбли: случайный лес и бустинг из лекции 5 обучаются в браузере "
            "заметно дольше, а нового о сложности модели не добавляют.\n"
            f"- Не нормирует штраф: сетка alpha та же, что в лекции, — от {_alpha_label(ALPHAS[0])} "
            f"до {_alpha_label(ALPHAS[-1])}, и её неодинаковый смысл для Ridge и Lasso показан как есть."
        )


if __name__ == "__main__":
    main()
