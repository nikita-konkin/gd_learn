"""Streamlit interface for the metric playground.

Three tabs. The first shows the trap: always answering "no failure" scores high
accuracy on rare-event data and finds nothing. The second moves the threshold
of lecture 4's model and prices its two kinds of mistake. The third puts
lecture 6's three remedies for imbalance side by side under one threshold.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import streamlit as st

from metric_playground.metrics import (
    REMEDIES,
    average_precision,
    cheapest_threshold,
    confusion,
    load_lecture4,
    load_lecture6,
    majority,
    nearest_recall,
    populations,
    precision_recall_curve,
)
from metric_playground.plotting import confusion_figure, curve_figure
from playground_common.links import other_playgrounds
from playground_common.wording import as_printed

TITLE = "Метрика и дисбаланс"

POPULATION_LABEL = "Набор данных"
THRESHOLD_LABEL = "Порог модели лекции 4"
MISS_COST_LABEL = "Пропуск отказа дороже ложной тревоги, раз"
REMEDY_THRESHOLD_LABEL = "Порог для моделей лекции 6"

# The recalls lecture 4 asks for, and the threshold scikit-learn uses by default.
LECTURE_RECALLS = (0.6, 0.8, 0.95)
DEFAULT_THRESHOLD = 0.5
DEFAULT_MISS_COST = 10


@lru_cache(maxsize=1)
def _lecture4():
    return load_lecture4()


@lru_cache(maxsize=1)
def _lecture6():
    return load_lecture6()


@lru_cache(maxsize=1)
def _populations():
    return populations()


@lru_cache(maxsize=32)
def _cheapest(miss: int) -> tuple[float, float]:
    labels, scores = _lecture4()
    return cheapest_threshold(labels, scores, miss)


def _controls() -> tuple[str, float, int, float]:
    """The sidebar: the data set to inspect, then each tab's own threshold, then the price of a miss."""
    names = [population.name for population in _populations()]
    st.sidebar.header("Ловушка")
    population = st.sidebar.selectbox(POPULATION_LABEL, names, index=0)

    st.sidebar.header("Порог")
    st.sidebar.caption("Порог меняет только решение модели; цена — только оценку решения.")
    threshold = st.sidebar.slider(THRESHOLD_LABEL, 0.0, 1.0, DEFAULT_THRESHOLD, step=0.01)
    miss = st.sidebar.slider(MISS_COST_LABEL, 1, 100, DEFAULT_MISS_COST)

    st.sidebar.header("Перекос классов")
    remedy_threshold = st.sidebar.slider(REMEDY_THRESHOLD_LABEL, 0.05, 0.95, DEFAULT_THRESHOLD, step=0.05)

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/metric"))
    return population, threshold, miss, remedy_threshold


def _trap_tab(name: str) -> None:
    population = next(item for item in _populations() if item.name == name)
    st.markdown(
        f"{population.name}: {population.positive} положительных из {population.total} "
        f"({population.source}). Модель отвечает «отказа нет» всегда."
    )
    left, middle, right, last = st.columns(4)
    left.metric("Доля правильных", f"{population.majority_accuracy * 100:.1f} %".replace(".", ","))
    middle.metric("Найдено отказов", "0")
    right.metric("Полнота", "0.000")
    last.metric("PR-AUC случайной модели", f"{population.pr_auc_baseline:.3f}")

    if population.name.startswith("Лекция 4"):
        labels, _ = _lecture4()
        counts = majority(labels)
        # Code spans: Streamlit's markdown would otherwise turn the lecture's "<-" into an arrow.
        st.info(
            f"Лекция 4 печатает `Доля отказов в выборке: {labels.mean():.1%}` и `Доля правильных ответов: "
            f"{counts.accuracy:.3f}  <- выглядит отлично`, а полноту, точность и F1 — нулевыми. Здесь то же самое."
        )
    elif population.name.startswith("SECOM"):
        st.info(
            "Это базовый уровень из задания на РГР: «93,4 % — столько даёт ответ «годен» всегда», а PR-AUC "
            "случайной модели — 0,066. Критерий успеха РГР должен быть выше обоих, иначе модель ничего не умеет."
        )
    st.caption(
        "Доля правильных растёт вместе с перекосом классов, а найдено по-прежнему ноль. Поэтому для редких "
        "событий метрику выбирают до обучения — по полноте и точности, а не по доле правильных."
    )


def _threshold_tab(threshold: float, miss: int) -> None:
    labels, scores = _lecture4()
    counts = confusion(labels, scores, threshold)
    st.markdown(
        "Лекция 4: логистическая регрессия с весами классов, отложенная выборка из "
        f"{counts.total} объектов, из них {int(labels.sum())} отказов."
    )

    left, middle, right = st.columns(3)
    left.metric("Полнота", as_printed(counts.recall))
    middle.metric("Точность", as_printed(counts.precision))
    right.metric("Доля правильных", as_printed(counts.accuracy))

    figure_column, curve_column = st.columns([2, 3])
    with figure_column:
        st.plotly_chart(confusion_figure(counts), use_container_width=True)
    with curve_column:
        precision, recall, _ = precision_recall_curve(labels, scores)
        st.plotly_chart(
            curve_figure(
                {"model": ("логистическая регрессия", precision, recall)},
                {"model": (counts.recall, counts.precision)},
                float(labels.mean()),
            ),
            use_container_width=True,
        )

    st.subheader("Что печатает лекция")
    rows = []
    points = [nearest_recall(labels, scores, wanted) for wanted in LECTURE_RECALLS]
    for point in points:
        rows.append(f"| {point.wanted:.2f} | {point.recall:.2f} | {point.precision:.2f} | {point.threshold:.3f} |")
    last = points[-1]
    alarms = (1 - last.precision) / last.precision
    st.markdown("| нужна полнота | получена | точность | порог |\n|---|---|---|---|\n" + "\n".join(rows))
    st.caption(
        f"PR-AUC модели — {average_precision(labels, scores):.3f}; от порога она не зависит. Каждая строка — "
        f"одна точка той же кривой: полноту {last.recall:.2f} модель покупает ценой точности "
        f"{last.precision:.2f}, то есть примерно {alarms:.0f} ложными тревогами на каждый найденный отказ."
    )

    st.subheader("Сколько стоит ошибка")
    best_threshold, best_cost = _cheapest(miss)
    best = confusion(labels, scores, best_threshold)
    current_cost = counts.cost(miss)
    left, middle, right = st.columns(3)
    left.metric("Стоимость при выбранном пороге", f"{current_cost:.0f}")
    middle.metric("Самый дешёвый порог", f"{best_threshold:.3f}")
    right.metric("Его стоимость", f"{best_cost:.0f}")
    st.caption(
        f"Пропуск отказа стоит {miss} ложных тревог. При самом дешёвом пороге модель пропускает "
        f"{best.false_negative} и поднимает {best.false_positive} ложных тревог. Измените цену — сместится "
        "и порог: выбор компромисса, как сказано в лекции, — «вопрос не математики, а предметной области»."
    )


def _remedies_tab(threshold: float) -> None:
    labels, scores = _lecture6()
    st.markdown(
        f"Лекция 6: случайный лес на данных с {labels.mean():.2%} редкого класса, три способа обучения, "
        f"отложенная выборка — {len(labels)} объектов, {int(labels.sum())} редких."
    )
    curves, points, rows = {}, {}, []
    for key, name in REMEDIES.items():
        counts = confusion(labels, scores[key], threshold)
        area = average_precision(labels, scores[key])
        precision, recall, _ = precision_recall_curve(labels, scores[key])
        curves[key] = (name, precision, recall)
        points[key] = (counts.recall, counts.precision)
        rows.append(f"| {name} | {area:.3f} | {counts.recall:.3f} | {counts.precision:.3f} |")
    st.markdown("| способ | PR-AUC | полнота | точность |\n|---|---|---|---|\n" + "\n".join(rows))
    st.plotly_chart(curve_figure(curves, points, float(labels.mean())), use_container_width=True)

    if np.isclose(threshold, DEFAULT_THRESHOLD):
        st.info("Порог 0.5 — тот, что в лекции 6. Таблица повторяет её вывод строка в строку.")
        plain = confusion(labels, scores["plain"], threshold)
        weighted = confusion(labels, scores["weighted"], threshold)
        if weighted.recall < plain.recall:
            st.warning(
                f"Веса классов здесь не подняли полноту, а снизили её: {as_printed(weighted.recall)} против "
                f"{as_printed(plain.recall)} без них. Лекция объясняет почему: деревья леса выращены до чистых "
                "листьев, а вероятность в чистом листе от весов не зависит — веса меняют лишь выбор разбиений."
            )
    st.caption(
        "PR-AUC у трёх способов различается в третьем знаке: ранжируют объекты они почти одинаково. "
        "Главное, что меняется, — где окажется рабочая точка при пороге 0.5. Подвиньте порог и посмотрите, "
        "как полнота и точность каждого способа расходятся в разные стороны, а PR-AUC стоит на месте."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Когда отказов два процента, модель, не нашедшая ни одного, верна в 98 случаях из 100. "
        "Здесь видно, какие метрики это скрывают и как порог превращает оценки модели в решения."
    )

    population, threshold, miss, remedy_threshold = _controls()
    trap, decision, remedies = st.tabs(["Ловушка доли правильных", "Порог и цена ошибки", "Как выправить перекос"])
    with trap:
        _trap_tab(population)
    with decision:
        _threshold_tab(threshold, miss)
    with remedies:
        _remedies_tab(remedy_threshold)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не обучает модели в браузере: оценки моделей лекций 4 и 6 посчитаны заранее, здесь по ним "
            "считаются только метрики. Поэтому страница не тянет scikit-learn и грузится быстро.\n"
            "- Не работает с набором AI4I 2020 из лабораторной модуля 2: модель там своя у каждого студента, "
            "и общих чисел для сверки нет. Механизм тот же.\n"
            "- SECOM представлен только числом браков: для ловушки доли правильных больше ничего не нужно.\n"
            "- Цена ошибки задана одним отношением; в реальной задаче стоимости приходят из предметной области "
            "и могут зависеть от объекта."
        )


if __name__ == "__main__":
    main()
