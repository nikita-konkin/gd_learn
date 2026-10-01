"""Streamlit interface for the network playground.

A 2-16-1 network trains on two moons in twenty lines of NumPy, and a
numerical check compares its gradients with central differences. This page
lets the student break one line of the backward pass and watch two things side by side: training, which keeps
looking plausible, and the gradient check, which does not.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import streamlit as st

from nn_playground.network import (
    BUGS,
    NO_BUG,
    RELU_BUG,
    TOLERANCE,
    Component,
    Training,
    gradient_check,
    load_moons,
    train,
)
from nn_playground.plotting import boundary_figure, history_figure
from playground_common.links import other_playgrounds
from playground_common.wording import as_printed, plural

TITLE = "Нейросеть на NumPy"

BUG_LABEL = "Ошибка в обратном проходе"
RATE_LABEL = "Скорость обучения"
EPOCHS_LABEL = "Эпох"
HIDDEN_LABEL = "Нейронов в скрытом слое"
SEED_LABEL = "Зерно инициализации весов"

RATES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
WIDTHS = [2, 4, 8, 16, 32, 64]

# Settings at which the correct network learns the moons well and quickly.
DEFAULTS = {"hidden": 16, "learning_rate": 0.5, "epochs": 400, "seed": 0}


@lru_cache(maxsize=1)
def _data():
    return load_moons()


@lru_cache(maxsize=64)
def _train(hidden: int, learning_rate: float, epochs: int, seed: int, bug: str) -> Training:
    train_x, test_x, train_y, test_y = _data()
    return train(train_x, train_y, test_x, test_y, hidden, learning_rate, epochs, seed, bug)


@lru_cache(maxsize=8)
def _check(bug: str) -> tuple[Component, ...]:
    train_x, _, train_y, _ = _data()
    return gradient_check(train_x, train_y, bug)


def _controls() -> tuple[str, int, float, int, int]:
    """The sidebar. The formula, the optimiser and the architecture each get their own controls."""
    st.sidebar.header("Формулы")
    st.sidebar.caption("Эта ручка меняет одну строку обратного прохода.")
    bug = st.sidebar.selectbox(BUG_LABEL, list(BUGS), index=0)

    st.sidebar.header("Обучение")
    st.sidebar.caption("Эти ручки меняют только спуск.")
    learning_rate = st.sidebar.select_slider(RATE_LABEL, RATES, value=DEFAULTS["learning_rate"])
    epochs = st.sidebar.slider(EPOCHS_LABEL, 50, 1000, DEFAULTS["epochs"], step=50)

    st.sidebar.header("Сеть")
    st.sidebar.caption("Эти ручки меняют только начальную сеть.")
    hidden = st.sidebar.select_slider(HIDDEN_LABEL, WIDTHS, value=DEFAULTS["hidden"])
    seed = st.sidebar.slider(SEED_LABEL, 0, 10, DEFAULTS["seed"])

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/nn"))
    return bug, hidden, learning_rate, epochs, seed


def _training_section(run: Training, reference: Training | None) -> None:
    left, middle, right = st.columns(3)
    left.metric("Потери до обучения", f"{run.initial_loss:.4f}")
    middle.metric("Потери после", f"{run.final_loss:.4f}")
    right.metric("Точность на отложенной выборке", as_printed(run.test_accuracy))

    if reference is not None:
        st.caption(
            f"Те же настройки без ошибки дают точность {as_printed(reference.test_accuracy)} "
            f"и потери {reference.final_loss:.4f}. Пунктир на графике — их кривая."
        )

    curve, boundary = st.columns(2)
    with curve:
        st.plotly_chart(history_figure(run.history, reference.history if reference else None), use_container_width=True)
    with boundary:
        _, test_x, _, test_y = _data()
        st.plotly_chart(boundary_figure(run.params, test_x, test_y), use_container_width=True)


def _check_section(components: tuple[Component, ...], bug: str) -> None:
    st.subheader("Численная проверка градиента")
    first = components[0]
    wrong = [component for component in components if not component.agrees]
    worst = max(component.relative_error for component in components)

    left, middle, right = st.columns(3)
    left.metric("Аналитический W1[0, 0]", f"{first.analytic:.6f}")
    middle.metric("Численный W1[0, 0]", f"{first.numeric:.6f}")
    right.metric("Расходятся компонент", f"{len(wrong)} из {len(components)}")

    if wrong:
        st.error(
            f"Формулы обратного прохода неверны: наибольшее относительное расхождение {worst:.2g} "
            f"при допуске {TOLERANCE:g}."
        )
    else:
        st.success(f"Все {len(components)} компонент совпадают: наибольшее относительное расхождение {worst:.1e}.")

    if bug == NO_BUG:
        st.caption(
            "Обычно сравнивают одну компоненту, например W1[0, 0]. Здесь проверены все "
            f"{len(components)} {plural(len(components), 'вес', 'веса', 'весов')} проверочной сети 2–5–1 "
            "на первых пятидесяти строках."
        )
    elif bug == RELU_BUG:
        untouched = len(components) - len(wrong)
        st.caption(
            "Самая коварная из трёх ошибок. Обучение идёт, точность выглядит правдоподобно, и ничто снаружи "
            "не подсказывает, что формулы неверны: внешне это выглядит как неудачные данные. "
            f"Проверка градиента её ловит — но не на всех весах: {untouched} весов "
            "второго слоя от производной ReLU не зависят и совпадают. Проверка одной компоненты "
            "из W2 вместо W1[0, 0] пропустила бы ошибку."
        )

    table = pd.DataFrame(
        {
            "вес": [component.name for component in components],
            "по формуле": [f"{component.analytic:.6f}" for component in components],
            "численно": [f"{component.numeric:.6f}" for component in components],
            "расхождение": [f"{component.relative_error:.1e}" for component in components],
            "совпадает": ["да" if component.agrees else "нет" for component in components],
        }
    )
    with st.expander(f"Все {len(components)} компонент"):
        st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Двухслойная сеть: прямой проход, потери, градиенты по правилу цепочки и шаг спуска — "
        "без фреймворка. Здесь можно сломать одну строку обратного прохода и посмотреть, кто это заметит."
    )

    bug, hidden, learning_rate, epochs, seed = _controls()
    run = _train(hidden, learning_rate, epochs, seed, bug)
    reference = _train(hidden, learning_rate, epochs, seed, NO_BUG) if bug != NO_BUG else None
    _training_section(run, reference)
    _check_section(_check(bug), bug)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не использует мини-пакеты: каждый шаг считается по всей обучающей выборке.\n"
            "- Не использует оптимизаторы сложнее простого градиентного спуска: ни импульса, ни Adam. "
            "Их удобнее пробовать во фреймворке, например в PyTorch.\n"
            "- Не меняет данные: два полумесяца подготовлены заранее и одни и те же при любых "
            "настройках.\n"
            "- Проверка градиента всегда идёт на маленькой проверочной сети 2–5–1, а не на обучаемой: "
            "ошибка в формуле от размера сети не зависит."
        )


if __name__ == "__main__":
    main()
