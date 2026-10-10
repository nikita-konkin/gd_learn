"""Streamlit interface for the metric playground.

Three tabs. The first shows the trap: always answering "no failure" scores high
accuracy on rare-event data and finds nothing. The second moves the threshold
of a failure detector and prices its two kinds of mistake. The third puts
three remedies for imbalance side by side under one threshold.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import streamlit as st

from metric_playground.metrics import (
    FAILURES,
    REMEDIES,
    SECOM,
    average_precision,
    cheapest_threshold,
    confusion,
    load_failures,
    load_rare_class,
    majority,
    nearest_recall,
    populations,
    precision_recall_curve,
)
from metric_playground.plotting import confusion_figure, curve_figure
from playground_common.explain import how_to_read, quick_start, reference
from playground_common.links import other_playgrounds
from playground_common.wording import as_printed

TITLE = "Метрика и дисбаланс"

POPULATION_LABEL = "Набор данных"
THRESHOLD_LABEL = "Порог: с какой уверенности объявлять отказ"
MISS_COST_LABEL = "Пропуск отказа дороже ложной тревоги, раз"
REMEDY_THRESHOLD_LABEL = "Порог для трёх способов"

# Three recalls worth asking for, and the threshold scikit-learn uses by default.
WANTED_RECALLS = (0.6, 0.8, 0.95)
DEFAULT_THRESHOLD = 0.5
DEFAULT_MISS_COST = 10


@lru_cache(maxsize=1)
def _failures():
    return load_failures()


@lru_cache(maxsize=1)
def _rare_class():
    return load_rare_class()


@lru_cache(maxsize=1)
def _populations():
    return populations()


@lru_cache(maxsize=32)
def _cheapest(miss: int) -> tuple[float, float]:
    labels, scores = _failures()
    return cheapest_threshold(labels, scores, miss)


def _controls() -> tuple[str, float, int, float]:
    """The sidebar: the data set to inspect, then each tab's own threshold, then the price of a miss."""
    names = [population.name for population in _populations()]
    st.sidebar.header("Ловушка")
    population = st.sidebar.selectbox(
        POPULATION_LABEL,
        names,
        index=0,
        help="Три выборки с разной долей редкого класса — от двух процентов до шести. "
        "Чем реже событие, тем сильнее ловушка.",
    )

    st.sidebar.header("Порог")
    st.sidebar.caption("Порог меняет только решение модели; цена — только оценку решения.")
    threshold = st.sidebar.slider(
        THRESHOLD_LABEL, 0.0, 1.0, DEFAULT_THRESHOLD, step=0.01,
        help="Модель выдаёт не «да/нет», а число от 0 до 1 — свою уверенность. Порог "
        "превращает это число в решение. Ниже порог — больше найденных отказов и больше "
        "ложных тревог; выше — наоборот.",
    )
    miss = st.sidebar.slider(
        MISS_COST_LABEL, 1, 100, DEFAULT_MISS_COST,
        help="Во сколько раз пропустить настоящий отказ дороже, чем зря поднять тревогу. "
        "Это число приходит из предметной области, а не из данных — и именно оно решает, "
        "какой порог считать лучшим.",
    )

    st.sidebar.header("Перекос классов")
    remedy_threshold = st.sidebar.slider(
        REMEDY_THRESHOLD_LABEL, 0.05, 0.95, DEFAULT_THRESHOLD, step=0.05,
        help="Отдельный порог для третьей вкладки: там три способа обучения сравниваются при одном и том же пороге.",
    )

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
    left.metric(
        "Доля правильных",
        f"{population.majority_accuracy * 100:.1f} %".replace(".", ","),
        help="Сколько ответов из ста верны. Модель, которая всегда отвечает «отказа нет», "
        "права почти всегда — просто потому, что отказы редки.",
    )
    middle.metric("Найдено отказов", "0", help="Ровно то, зачем модель и строили.")
    right.metric(
        "Полнота",
        "0.000",
        help="Какая доля настоящих отказов найдена. Здесь ноль: не найден ни один.",
    )
    last.metric(
        "PR-AUC случайной модели",
        f"{population.pr_auc_baseline:.3f}",
        help="Уровень, который даёт слепое угадывание. Он равен доле редкого класса. "
        "Любая оценка модели ниже этого числа означает, что модель хуже случайной.",
    )

    if population.name == FAILURES:
        labels, _ = _failures()
        counts = majority(labels)
        st.info(
            f"Отказов в выборке {labels.mean():.1%}, доля правильных ответов {counts.accuracy:.3f} — выглядит "
            "отлично. А полнота, точность и F1 нулевые: модель не нашла ни одного отказа."
        )
    elif population.name == SECOM:
        st.info(
            "Это базовый уровень из задания на РГР: «93,4 % — столько даёт ответ «годен» всегда», а PR-AUC "
            "случайной модели — 0,066. Критерий успеха РГР должен быть выше обоих, иначе модель ничего не умеет."
        )
    st.caption(
        "Доля правильных растёт вместе с перекосом классов, а найдено по-прежнему ноль. Поэтому для редких "
        "событий метрику выбирают до обучения — по полноте и точности, а не по доле правильных."
    )


def _threshold_tab(threshold: float, miss: int) -> None:
    labels, scores = _failures()
    counts = confusion(labels, scores, threshold)
    st.markdown(
        "Логистическая регрессия с весами классов, отложенная выборка из "
        f"{counts.total} объектов, из них {int(labels.sum())} отказов."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Полнота",
        as_printed(counts.recall),
        help="Из всех настоящих отказов — какую долю модель нашла. Растёт, когда порог опускают.",
    )
    middle.metric(
        "Точность",
        as_printed(counts.precision),
        help="Из всех поднятых тревог — какая доля оказалась настоящим отказом. "
        "Падает, когда порог опускают. Поднять обе сразу нельзя.",
    )
    right.metric(
        "Доля правильных",
        as_printed(counts.accuracy),
        help="Сколько ответов верны вообще. На редких событиях это число почти ни о чём "
        "не говорит — смотрите на два предыдущих.",
    )

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

    st.subheader("Нужная полнота и её цена")
    rows = []
    points = [nearest_recall(labels, scores, wanted) for wanted in WANTED_RECALLS]
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
    left.metric(
        "Стоимость при выбранном пороге",
        f"{current_cost:.0f}",
        help="Пропуски и ложные тревоги, сложенные с учётом их цены. Единица — одна "
        "ложная тревога.",
    )
    middle.metric(
        "Самый дешёвый порог",
        f"{best_threshold:.3f}",
        help="Тот порог, при котором эта сумма минимальна. Он зависит от цены ошибки, "
        "а не от самой модели.",
    )
    right.metric("Его стоимость", f"{best_cost:.0f}")
    st.caption(
        f"Пропуск отказа стоит {miss} ложных тревог. При самом дешёвом пороге модель пропускает "
        f"{best.false_negative} и поднимает {best.false_positive} ложных тревог. Измените цену — сместится "
        "и порог: выбор компромисса — вопрос не математики, а предметной области."
    )


def _remedies_tab(threshold: float) -> None:
    labels, scores = _rare_class()
    st.markdown(
        f"Случайный лес на данных с {labels.mean():.2%} редкого класса, три способа обучения, "
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
        st.info("Порог 0.5 — тот, что scikit-learn использует в predict по умолчанию.")
        plain = confusion(labels, scores["plain"], threshold)
        weighted = confusion(labels, scores["weighted"], threshold)
        if weighted.recall < plain.recall:
            st.warning(
                f"Веса классов здесь не подняли полноту, а снизили её: {as_printed(weighted.recall)} против "
                f"{as_printed(plain.recall)} без них. Причина в устройстве леса: деревья выращены до чистых "
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

    quick_start(
        st,
        purpose=(
            "Когда событий одного класса единицы, доля верных ответов обманывает. Страница — про выбор метрики "
            "и порога при перекошенных классах."
        ),
        steps=(
            "Первая вкладка: доля правильных — 97,9 %, а найдено отказов — 0.",
            "Вторая: двигайте **«Порог: с какой уверенности объявлять отказ»** и следите за полнотой и "
            "точностью.",
            "Поменяйте **«Пропуск отказа дороже ложной тревоги, раз»**: самый дешёвый порог сдвинется.",
        ),
        notice="полноту и точность сразу не поднять; порог выбирают по цене ошибок, а не по одной метрике.",
    )
    how_to_read(
        st,
        what=(
            "Страница про выбор метрики, когда классы перекошены: событий одного рода много, "
            "другого — считанные единицы. Отказ оборудования, брак на производстве, мошенничество — "
            "все такие задачи выглядят одинаково. «Доля верных ответов» в них бесполезна: модель, "
            "которая всегда отвечает «всё хорошо», права в 98 случаях из 100 и при этом не находит "
            "ровно ничего. Три вкладки: сама ловушка, затем порог и цена двух видов ошибки, затем "
            "три способа учесть перекос."
        ),
        controls=(
            "**Набор данных** — для первой вкладки; меняет, насколько редок редкий класс. "
            "**Порог** — главная ручка второй вкладки: модель выдаёт не ответ, а число от 0 до 1, "
            "и порог превращает это число в решение. **Цена пропуска** — во сколько раз пропустить "
            "отказ дороже ложной тревоги. **Порог для трёх способов** — отдельная ручка третьей "
            "вкладки."
        ),
        watch=(
            "На первой вкладке сравните два числа: доля правильных около 98 % и найдено отказов — "
            "ноль. На второй двигайте порог и следите, как полнота и точность расходятся в разные "
            "стороны: поднять обе сразу невозможно, и это не недостаток модели, а свойство задачи. "
            "Таблица «нужная полнота и её цена» переводит это в рабочие слова: чтобы находить 95 % "
            "отказов, придётся смириться с таким-то числом ложных тревог на каждый найденный. "
            "Поменяйте цену пропуска — и самый дешёвый порог сдвинется, хотя модель осталась той же."
        ),
        data=(
            "Оценки моделей посчитаны заранее на отложенных выборках и загружены готовыми — "
            "поэтому страница быстрая. SECOM — реальный набор данных о браке на производстве "
            "полупроводников; числа по нему те же, что в задании на РГР."
        ),
    )
    reference(
        st,
        "imbalance",
        "accuracy",
        "baseline",
        "confusion_matrix",
        "precision_recall",
        "f1",
        "threshold",
        "pr_auc",
        "error_cost",
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
            "- Не обучает модели в браузере: оценки моделей посчитаны заранее, здесь по ним "
            "считаются только метрики. Поэтому страница не тянет scikit-learn и грузится быстро.\n"
            "- Не работает с набором AI4I 2020: модель на нём своя у каждого студента. Механизм тот же.\n"
            "- SECOM представлен только числом браков: для ловушки доли правильных больше ничего не нужно.\n"
            "- Цена ошибки задана одним отношением; в реальной задаче стоимости приходят из предметной области "
            "и могут зависеть от объекта."
        )


if __name__ == "__main__":
    main()
