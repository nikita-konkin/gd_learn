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
    SIGN_BUG,
    SIZE_BUG,
    TOLERANCE,
    Component,
    Training,
    gradient_check,
    load_moons,
    train,
)
from nn_playground.plotting import boundary_figure, history_figure
from playground_common.explain import how_to_read, quick_start
from playground_common.explain import reference as glossary
from playground_common.links import other_playgrounds
from playground_common.wording import as_printed, plural

TITLE = "Нейросеть на NumPy"

BUG_LABEL = "Сломать одну строку обратного прохода"
RATE_LABEL = "Размер шага (скорость обучения)"
EPOCHS_LABEL = "Эпох — проходов по всем данным"
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
    st.sidebar.caption("Эта настройка меняет одну строку обратного прохода.")
    bug = st.sidebar.selectbox(
        BUG_LABEL, list(BUGS), index=0,
        help=(
            "Главная настройка страницы. Каждый вариант портит ровно одну строку в формулах градиента — так, как "
            "это обычно случается в собственном коде."
        ),
    )

    st.sidebar.header("Обучение")
    st.sidebar.caption("Эти настройки меняют только спуск.")
    learning_rate = st.sidebar.select_slider(
        RATE_LABEL, RATES, value=DEFAULTS["learning_rate"],
        help="Насколько сильно сдвигать веса за один шаг. При слишком малом шаге сеть не успевает "
        "обучиться, при слишком большом потери колеблются и растут.",
    )
    epochs = st.sidebar.slider(
        EPOCHS_LABEL, 50, 1000, DEFAULTS["epochs"], step=50,
        help="Одна эпоха — один проход по всей обучающей выборке. Здесь каждая эпоха "
        "даёт ровно один шаг спуска: мини-пакетов нет.",
    )

    st.sidebar.header("Сеть")
    st.sidebar.caption("Эти настройки меняют только начальную сеть.")
    hidden = st.sidebar.select_slider(
        HIDDEN_LABEL, WIDTHS, value=DEFAULTS["hidden"],
        help="Сколько нейронов в скрытом слое. Двух почти наверняка мало для двух "
        "полумесяцев, шестидесяти четырёх заведомо достаточно.",
    )
    seed = st.sidebar.slider(
        SEED_LABEL, 0, 10, DEFAULTS["seed"],
        help="Веса сети перед обучением заполняются случайными числами. Зерно определяет, "
        "какими именно, а от начальной точки зависит результат.",
    )

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/nn"))
    return bug, hidden, learning_rate, epochs, seed


# The backward pass as the page prints it, one line per gradient; a bug swaps exactly one line.
BACKWARD = (
    ("out", "d_out = (p - y) / n               # ошибка выхода, усреднённая по n точкам"),
    ("", "d_W2 = hidden.T @ d_out"),
    ("", "d_hidden = d_out @ W2.T"),
    ("relu", "d_raw = d_hidden * (raw > 0)      # производная ReLU: 1, где нейрон был активен"),
    ("", "d_W1 = x.T @ d_raw"),
    ("", "W1 -= rate * d_W1; W2 -= rate * d_W2  # шаг спуска против градиента"),
)
BROKEN = {
    SIZE_BUG: ("out", "d_out = (p - y)                   # ← ОШИБКА: не поделено на n"),
    SIGN_BUG: ("out", "d_out = (y - p) / n               # ← ОШИБКА: знак перепутан"),
    RELU_BUG: ("relu", "d_raw = d_hidden                  # ← ОШИБКА: производная ReLU забыта"),
}


def _network_section(hidden: int, bug: str) -> None:
    """What the network is and which line of its gradient the chosen bug breaks."""
    st.subheader("1. Сеть и задача")
    st.markdown(
        "Задача — отнести точку плоскости к одному из двух классов «двух полумесяцев». Сеть получает две "
        f"координаты точки, передаёт их в скрытый слой из {hidden} "
        f"{plural(hidden, 'нейрона', 'нейронов', 'нейронов')} с функцией ReLU (отрицательное обнуляется, "
        "положительное проходит без изменений), а выходной нейрон с сигмоидой выдаёт вероятность класса 1. "
        "Ошибку сети измеряют потерями — перекрёстной энтропией: чем увереннее сеть ошибается, тем они больше."
    )
    st.markdown(
        "Обучение повторяет четыре шага: **прямой проход** (вычислить ответ), **потери**, **обратный "
        "проход** (вычислить по правилу цепочки, как потери зависят от каждого веса, т. е. градиент) и "
        "**шаг спуска** (сдвинуть веса против градиента). Ниже — обратный проход и шаг спуска в той форме, "
        "в какой их пишут вручную; выбранная слева ошибка заменяет ровно одну строку."
    )
    place, replacement = BROKEN.get(bug, ("", ""))
    lines = (replacement if place and key == place else line for key, line in BACKWARD)
    st.code("\n".join(lines), language="python")


def _training_section(run: Training, reference: Training | None) -> None:
    st.subheader("2. Обучение")
    left, middle, right = st.columns(3)
    left.metric(
        "Потери до обучения",
        f"{run.initial_loss:.4f}",
        help="Ошибка необученной сети со случайными весами; с ней сравнивается результат.",
    )
    middle.metric(
        "Потери после обучения",
        f"{run.final_loss:.4f}",
        help="Та же ошибка после всех эпох. Именно её уменьшает градиентный спуск.",
    )
    right.metric(
        "Доля правильных ответов на отложенной выборке",
        as_printed(run.test_accuracy),
        help="Считается по точкам, которых сеть не видела при обучении.",
    )

    if reference is not None:
        st.caption(
            f"Те же настройки без ошибки дают долю правильных ответов {as_printed(reference.test_accuracy)} "
            f"и потери {reference.final_loss:.4f}. Пунктир на графике — их кривая."
        )

    st.markdown(
        "Слева — потери на каждой эпохе: у исправной сети они быстро падают. Горизонтальная линия ln 2 — "
        "потери сети, которая для любой точки отвечает «вероятность 0.5». Справа — точки отложенной "
        "выборки на фоне областей, которые сеть относит к каждому классу."
    )
    curve, boundary = st.columns(2)
    with curve:
        st.plotly_chart(history_figure(run.history, reference.history if reference else None), use_container_width=True)
    with boundary:
        _, test_x, _, test_y = _data()
        st.plotly_chart(boundary_figure(run.params, test_x, test_y), use_container_width=True)


def _check_section(components: tuple[Component, ...], bug: str) -> None:
    st.subheader("3. Численная проверка градиента")
    first = components[0]
    wrong = [component for component in components if not component.agrees]
    worst = max(component.relative_error for component in components)

    st.caption(
        "Проверка устроена так: один вес немного сдвигают в обе стороны и смотрят, "
        "как при этом меняется ошибка. Получается тот же градиент, но найденный измерением, "
        "а не по формуле. Если формула верна, два числа совпадают почти до последнего знака."
    )
    left, middle, right = st.columns(3)
    left.metric(
        "По формуле, вес W1[0, 0]",
        f"{first.analytic:.6f}",
        help="Градиент, полученный обратным проходом, т. е. вашим кодом.",
    )
    middle.metric(
        "Измерением, тот же вес",
        f"{first.numeric:.6f}",
        help="Градиент, вычисленный сдвигом веса на малую величину. Это медленно, зато "
        "не зависит от формул.",
    )
    right.metric(
        "Расходящихся весов",
        f"{len(wrong)} из {len(components)}",
        help=f"Расхождением считается относительная разница больше {TOLERANCE:g}.",
    )

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
            "Самая коварная из трёх ошибок. Обучение идёт, доля правильных ответов правдоподобна, и ничто снаружи "
            "не указывает на неверные формулы: внешне это похоже на неудачные данные. "
            f"Проверка градиента её обнаруживает, но не на всех весах: {untouched} весов "
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

    quick_start(
        st,
        purpose=(
            "Показать на нейросети, обученной в браузере на NumPy, к чему приводит ошибка в "
            "формулах обратного прохода."
        ),
        steps=(
            "В **«Сломать одну строку обратного прохода»** выберите «забыта производная ReLU».",
            "Посмотрите на потери и долю правильных ответов: обучение идёт, и ошибки по ним не видно.",
            "Перейдите к численной проверке градиента ниже: расходятся 15 весов из 21.",
        ),
        notice="неверный градиент заметно не нарушает обучение; обнаружить его может только численная проверка.",
    )
    how_to_read(
        st,
        what=(
            "Нейросеть целиком, без PyTorch: двадцать строк NumPy. Два входа, шестнадцать "
            "нейронов в скрытом слое, один выход. Чтобы обучаться, сеть должна знать, как "
            "каждый её вес влияет на ошибку, — это и вычисляет обратный проход. Страница "
            "отвечает на практический вопрос: что будет, если в этих формулах ошибиться. "
            "Ответ неприятный: обучение всё равно пойдёт, графики будут выглядеть "
            "правдоподобно, и ничто снаружи не укажет на ошибку в коде. Обнаружить её "
            "может только численная проверка градиента внизу страницы."
        ),
        controls=(
            "**Сломать одну строку обратного прохода** — главная настройка: она определяет, какую строку "
            "формул испортить, или оставляет их верными. Три варианта ошибки — те, что "
            "чаще всего возникают при самостоятельной реализации. **Обучение** — размер "
            "шага и число эпох. **Сеть** — число нейронов в скрытом слое и начальные "
            "случайные веса."
        ),
        watch=(
            "Выберите «забыта производная ReLU». В верхней части страницы потери падают, граница на рисунке "
            "выглядит осмысленной, доля правильных ответов приемлема — по этим признакам ошибку не заметить. "
            "Ниже проверка градиента сразу показывает расхождение. Обратите внимание, "
            "сколько весов при этом совпадает: проверка одного веса из второго слоя пропустила "
            "бы ошибку, и именно так её обычно и пропускают. Пунктир на графике — "
            "та же сеть с верными формулами, для сравнения."
        ),
        data=(
            "«Два полумесяца» — стандартный учебный набор: два класса, которые нельзя "
            "разделить прямой. Данные одни и те же при любых настройках, обучение идёт "
            "в браузере по-настоящему."
        ),
    )
    glossary(
        st,
        "backprop",
        "gradient_check",
        "gradient_descent",
        "learning_rate",
        "loss",
        "epoch",
        "accuracy",
    )

    bug, hidden, learning_rate, epochs, seed = _controls()
    run = _train(hidden, learning_rate, epochs, seed, bug)
    reference = _train(hidden, learning_rate, epochs, seed, NO_BUG) if bug != NO_BUG else None
    _network_section(hidden, bug)
    _training_section(run, reference)
    _check_section(_check(bug), bug)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не использует мини-пакеты: каждый шаг считается по всей обучающей выборке.\n"
            "- Не использует оптимизаторы сложнее простого градиентного спуска: ни импульса, ни Adam. "
            "Их удобнее пробовать во фреймворке, например в PyTorch.\n"
            "- Не меняет данные: два полумесяца подготовлены заранее и одни и те же при любых "
            "настройках.\n"
            "- Не проверяет градиент на обучаемой сети: проверка всегда идёт на маленькой сети 2–5–1, "
            "потому что ошибка в формуле от размера сети не зависит."
        )


if __name__ == "__main__":
    main()
