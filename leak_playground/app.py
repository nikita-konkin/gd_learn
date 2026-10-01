"""Streamlit interface for the leakage playground.

The page answers one question: what does it cost to select features before
splitting the sample? With the default settings — 300 rows, 2000 columns of
pure noise — the leaky pipeline scores about 0.76 and the honest one about 0.51,
on data that holds nothing to predict.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from playground_common.compat import patch_pyarrow_stub

# Before the first sklearn import: stlite's pyarrow stub is missing types that
# sklearn's input check reaches for on every call.
patch_pyarrow_stub()

from leak_playground.experiment import (  # noqa: E402  — must follow the stub patch
    TRUTH,
    Scores,
    Selection,
    all_p_values,
    chance_overlap,
    expected_false_positives,
    scores,
    selection,
)
from leak_playground.noise import INFORMATIVE, NoiseSettings, make_data  # noqa: E402
from leak_playground.plotting import pvalues_figure, recurrence_figure, scores_figure  # noqa: E402
from playground_common.links import other_playgrounds  # noqa: E402
from playground_common.wording import as_printed, features, observations, plural  # noqa: E402

TITLE = "Утечка данных"

OBSERVATIONS_LABEL = "Наблюдений"
FEATURES_LABEL = "Признаков"
SIGNAL_LABEL = "Сила настоящего сигнала"
SEED_LABEL = "Зерно генерации данных"
SELECT_LABEL = "Отбираем признаков"
FOLDS_LABEL = "Блоков кросс-валидации"

# Wide pure noise and few rows: the setting where the leak is plainest.
DEFAULT = NoiseSettings()
DEFAULT_SELECT = 20
DEFAULT_FOLDS = 5


@lru_cache(maxsize=32)
def _data(settings: NoiseSettings):
    return make_data(settings)


@lru_cache(maxsize=64)
def _scores(settings: NoiseSettings, select: int, folds: int) -> Scores:
    matrix, labels = _data(settings)
    return scores(matrix, labels, select, folds)


@lru_cache(maxsize=64)
def _selection(settings: NoiseSettings, select: int, folds: int) -> Selection:
    matrix, labels = _data(settings)
    return selection(matrix, labels, select, folds)


@lru_cache(maxsize=32)
def _p_values(settings: NoiseSettings) -> tuple[float, ...]:
    matrix, labels = _data(settings)
    return all_p_values(matrix, labels)


def _controls() -> tuple[NoiseSettings, int, int]:
    """The sidebar. Every control decides exactly one thing.

    Three groups, kept apart on purpose: what the data is, how many features the
    selection keeps, and how the sample is cut into blocks. One control that
    moved two of those at once is what made the sample repository's seed slider
    useless — moving it changed everything, so nothing followed from the
    experiment.
    """
    st.sidebar.header("Данные")
    st.sidebar.caption("Эти ручки меняют только сами данные.")
    count = st.sidebar.slider(OBSERVATIONS_LABEL, 100, 600, DEFAULT.observations, step=50)
    width = st.sidebar.slider(FEATURES_LABEL, 200, 4000, DEFAULT.features, step=200)
    signal = st.sidebar.slider(SIGNAL_LABEL, 0.0, 1.0, DEFAULT.signal, step=0.05)
    seed = st.sidebar.slider(SEED_LABEL, 0, 20, DEFAULT.seed)

    st.sidebar.header("Отбор признаков")
    st.sidebar.caption("Эта ручка меняет только отбор.")
    select = st.sidebar.slider(SELECT_LABEL, 5, 100, DEFAULT_SELECT, step=5)

    st.sidebar.header("Проверка")
    st.sidebar.caption("Эта ручка меняет только разбиение на блоки.")
    folds = st.sidebar.slider(FOLDS_LABEL, 3, 10, DEFAULT_FOLDS)

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/leak"))

    return NoiseSettings(observations=count, features=width, seed=seed, signal=signal), select, folds


def _headline(result: Scores, settings: NoiseSettings, select: int, folds: int) -> None:
    left, middle, right = st.columns(3)
    left.metric("С утечкой", as_printed(result.leaky))
    middle.metric("Честно", as_printed(result.honest))
    right.metric("Разрыв", f"+{as_printed(result.gap)}")

    if settings.signal == 0:
        st.caption(
            f"Истина — {as_printed(TRUTH)}: классы сбалансированы, а признаки не связаны с целью никак. "
            f"Разрыв в {as_printed(result.gap)} придуман отбором, обученным на отложенных строках."
        )
    else:
        st.caption(
            f"Сигнал подмешан в {features(min(INFORMATIVE, settings.features))}, сила {settings.signal:.2f}. "
            "Честная оценка теперь выше 0.500 законно — а утечка по-прежнему добавляет сверху, "
            "и по одному числу эти две прибавки не различить."
        )


def _recurrence_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("Что именно отобралось")
    st.plotly_chart(recurrence_figure(selected), use_container_width=True)

    chance = chance_overlap(select, settings.features)
    survived = sum(1 for value in selected.recurrence if value == selected.folds)
    informative = min(INFORMATIVE, settings.features)
    found = sum(1 for index in selected.indices if index < informative)

    left, right = st.columns(2)
    left.metric(f"Переотобраны во всех {selected.folds} блоках", f"{survived} из {select}")
    if settings.signal:
        right.metric("Из них настоящих", f"{found} из {informative}")
    else:
        right.metric("Настоящих признаков в данных", "0")

    if settings.signal == 0:
        if survived:
            reached = f"а {survived} из {select} всё равно {plural(survived, 'дотянулся', 'дотянулись', 'дотянулись')}"
        else:
            reached = f"и ни один из {select} не дотянулся"
        st.caption(
            "Признак с настоящим сигналом переотбирается в каждом блоке. Высота столбца — "
            "это всё, что отбор может предъявить в своё оправдание, и доказательством она "
            f"не является: настоящих признаков здесь нет вовсе, {reached} до верха."
        )
    else:
        st.caption(
            f"Настоящий сигнал лежит в первых {informative} столбцах, и отбор нашёл {found} из них. "
            f"Во всех {selected.folds} блоках переотобрались {survived} из {select} признаков: "
            "переотбор отслеживает сигнал, но по нему одному настоящий признак от удачливого "
            "шумового не отличить."
        )

    st.caption(
        f"Два независимых отбора совпали бы по случайности в {chance:.2f} признака из {select}; "
        f"два блока здесь совпадают в {selected.mean_pairwise_overlap:.1f} в среднем — выше "
        "случайного, но лишь потому, что любые два блока делят между собой часть одних и тех же "
        "строк. Именно эта общая часть и просачивается в оценку."
    )

    table = pd.DataFrame(
        {
            "признак": [f"#{index}" for index in selected.indices],
            "F": [round(value, 2) for value in selected.f_scores],
            "p": [f"{value:.2g}" for value in selected.p_values],
            f"блоков из {selected.folds}": list(selected.recurrence),
        }
    )
    with st.expander(f"Таблица отобранных признаков ({select})"):
        st.dataframe(table, use_container_width=True, hide_index=True)


def _pvalues_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("Почему отбор нашёл «значимое»")
    st.plotly_chart(pvalues_figure(np.asarray(_p_values(settings))), use_container_width=True)

    expected = expected_false_positives(selected.tested)
    left, right = st.columns(2)
    left.metric("Признаков с p < 0.05", f"{selected.significant_at_05} из {selected.tested}")
    right.metric("Выдержали поправку Бонферрони", str(selected.significant_after_bonferroni))
    st.caption(
        f"Проверено {features(selected.tested)}, порог 0.05 — значит около {expected:.0f} пройдут его "
        f"по случайности. Прошли {selected.significant_at_05}. "
        f"Поправку на {selected.tested} проверок выдержали "
        f"{selected.significant_after_bonferroni} — и отбор всё равно взял лучшие "
        f"{select} и передал их модели."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Отбор признаков до разбиения выборок — самая частая и самая незаметная ошибка. "
        "Здесь она измерена на данных, в которых предсказывать нечего."
    )

    settings, select, folds = _controls()
    result = _scores(settings, select, folds)
    selected = _selection(settings, select, folds)

    _headline(result, settings, select, folds)
    st.caption(
        f"{observations(settings.observations)}, {features(settings.features)}, отобрано {select}, блоков {folds}."
    )

    st.plotly_chart(scores_figure(result.leaky_folds, result.honest_folds), use_container_width=True)
    st.caption(
        "Разрыв держится в каждом блоке, а не в среднем по счастливому разбиению. "
        "Обе ветви масштабируют признаки и обучают одну и ту же модель; различаются они "
        "только тем, где сделан отбор."
    )

    _recurrence_section(selected, settings, select)
    _pvalues_section(selected, settings, select)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не показывает другие виды утечки: признак, вычисленный из будущего, "
            "дубликаты между обучающей и отложенной частями, целевое кодирование без "
            "вложенной проверки. Механизм у них тот же, а данные нужны другие.\n"
            "- Не меняет модель: логистическая регрессия тут всюду. Утечка не зависит "
            "от выбора модели — это свойство порядка операций.\n"
            "- Не считает вложенную кросс-валидацию: подбор гиперпараметров на той же "
            "проверочной части — отдельная тема.\n"
            f"- Сигнал, когда он включён, кладётся всегда в первые {INFORMATIVE} столбцов "
            "и всегда как сдвиг среднего. Это не модель реальных данных, а способ увидеть, "
            "что по одному числу утечку от сигнала не отличить."
        )


if __name__ == "__main__":
    main()
