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
from playground_common.explain import how_to_read, quick_start, reference  # noqa: E402
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

# The corner of the table shown under «Как выглядит таблица».
PREVIEW_ROWS = 8
PREVIEW_COLUMNS = 8


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
    count = st.sidebar.slider(
        OBSERVATIONS_LABEL, 100, 600, DEFAULT.observations, step=50,
        help="Сколько строк в таблице. Чем их меньше относительно числа столбцов, "
        "тем легче найти в шуме мнимую закономерность.",
    )
    width = st.sidebar.slider(
        FEATURES_LABEL, 200, 4000, DEFAULT.features, step=200,
        help="Сколько столбцов-признаков. Все они, если сигнал выключен, — чистый шум: "
        "случайные числа, никак не связанные с тем, что предсказываем.",
    )
    signal = st.sidebar.slider(
        SIGNAL_LABEL, 0.0, 1.0, DEFAULT.signal, step=0.05,
        help=f"На сколько у объектов класса 1 в среднем больше значения в столбцах #0–#{INFORMATIVE - 1}. "
        "При нуле связи признаков с классом нет, предсказывать нечего, и любая оценка выше 0.500 "
        "целиком придумана. При 1 сдвиг равен разбросу шума — по этим столбцам классы различимы.",
    )
    seed = st.sidebar.slider(
        SEED_LABEL, 0, 20, DEFAULT.seed,
        help="Другое зерно — другой шум при тех же настройках. Разрыв не исчезает: "
        "он не случайность одного набора данных.",
    )

    st.sidebar.header("Отбор признаков")
    st.sidebar.caption("Эта ручка меняет только отбор.")
    select = st.sidebar.slider(
        SELECT_LABEL, 5, 100, DEFAULT_SELECT, step=5,
        help=(
            "Сколько самых «значимых» столбцов оставить модели. Именно этот шаг и совершает утечку, если сделан до "
            "разбиения выборки."
        ),
    )

    st.sidebar.header("Проверка")
    st.sidebar.caption("Эта ручка меняет только разбиение на блоки.")
    folds = st.sidebar.slider(
        FOLDS_LABEL, 3, 10, DEFAULT_FOLDS,
        help=(
            "На сколько частей делится выборка при проверке: каждая по очереди становится отложенной, остальные — "
            "обучающими."
        ),
    )

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/leak"))

    return NoiseSettings(observations=count, features=width, seed=seed, signal=signal), select, folds


def _data_section(settings: NoiseSettings, select: int, folds: int) -> None:
    """What the table is, what is predicted and what «signal» means, before any number about them."""
    informative = min(INFORMATIVE, settings.features)
    columns = f"#0–#{informative - 1}"
    if settings.signal:
        signal = (
            f"Сейчас сила {settings.signal:.2f}: у объектов класса 1 значения в столбцах {columns} в среднем на "
            f"{settings.signal:.2f} больше, чем у класса 0, — по этим {informative} столбцам классы действительно "
            f"можно различить. Остальные {settings.features - informative} — по-прежнему шум."
        )
    else:
        signal = (
            "Сейчас его нет: сила 0, признаки и классы независимы, предсказывать нечего. Если поднять "
            f"«{SIGNAL_LABEL}», скажем, до 0.6, у объектов класса 1 значения в столбцах {columns} станут в среднем "
            "на 0.6 больше, чем у класса 0, и по этим столбцам классы можно будет различить."
        )

    st.subheader("Данные и задача")
    st.markdown(
        f"- **Таблица** — {observations(settings.observations)} (строки) и {features(settings.features)} "
        f"(столбцы #0 … #{settings.features - 1}). Данные не реальные: каждое значение — случайное число "
        "из нормального распределения со средним 0 и разбросом 1, таблица генерируется прямо на странице. "
        "Поэтому правильный ответ известен заранее.\n"
        "- **Что предсказываем** — класс каждой строки, 0 или 1. Классы тоже раздаются случайно, примерно "
        "поровну. Модель — логистическая регрессия, оценка — доля верных ответов на кросс-валидации; "
        f"угадывание наугад даёт {as_printed(TRUTH)}.\n"
        f"- **Сигнал** — настоящая связь признаков с классом, то, что модели стоило бы найти. {signal}\n"
        f"- **Где утечка.** Перед обучением из {features(settings.features)} оставляют {select}, сильнее всего "
        "различающихся между классами. Если отобрать их один раз по всей таблице, отбор видит и строки, на "
        "которых модель потом проверяют, и находит «различия» даже в чистом шуме. Честно — повторять отбор "
        f"внутри каждого из {folds} блоков кросс-валидации, только по его обучающей части."
    )

    matrix, labels = _data(settings)
    shown = min(PREVIEW_COLUMNS, settings.features)
    preview = pd.DataFrame(matrix[:PREVIEW_ROWS, :shown].round(2), columns=[f"#{index}" for index in range(shown)])
    preview["класс"] = labels[:PREVIEW_ROWS]
    means = [float(matrix[labels == label, :informative].mean()) for label in (0, 1)]
    with st.expander("Как выглядит таблица"):
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.caption(
            f"Первые {PREVIEW_ROWS} строк и {shown} столбцов из {settings.features}. Среднее по столбцам "
            f"{columns}: у класса 0 — {means[0]:.2f}, у класса 1 — {means[1]:.2f}. Это и есть сигнал: при силе 0 "
            "средние почти совпадают, при силе s различаются примерно на s."
        )


def _headline(result: Scores, settings: NoiseSettings, select: int, folds: int) -> None:
    st.subheader("Две оценки одной модели")
    left, middle, right = st.columns(3)
    left.metric(
        "С утечкой",
        as_printed(result.leaky),
        help="Доля верных ответов, когда отбор признаков сделан один раз на всей таблице — "
        "то есть уже подсмотрев отложенные строки.",
    )
    middle.metric(
        "Честно",
        as_printed(result.honest),
        help="Та же модель, но отбор признаков делается заново внутри каждого блока, "
        "только по обучающей части. Это и есть правильный порядок.",
    )
    right.metric(
        "Разрыв",
        f"+{as_printed(result.gap)}",
        help="Разница между ними. Это цифра, на которую вы обманете себя и заказчика, "
        "если поставите отбор признаков не туда.",
    )

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
    informative = min(INFORMATIVE, settings.features)
    st.markdown(
        f"Отбор по всей таблице выбрал {select} столбцов — их номера внизу графика. Насколько этот выбор "
        f"надёжен? Повторим отбор {selected.folds} раз, как это делает честная ветвь: каждый раз только по "
        "обучающей части одного блока. Высота столбика — в скольких блоках признак отобран снова.\n\n"
        "Признак с настоящей связью выбирается почти при любом разбиении, и его столбик доходит до пунктира. "
        "Признак, попавший в выбор по случайному совпадению, на других строках уже не выглядит лучшим, и "
        "его столбик низкий."
    )
    st.plotly_chart(recurrence_figure(selected, informative if settings.signal else 0), use_container_width=True)

    chance = chance_overlap(select, settings.features)
    survived = sum(1 for value in selected.recurrence if value == selected.folds)
    found = sum(1 for index in selected.indices if index < informative)

    left, right = st.columns(2)
    left.metric(f"Отобраны снова во всех {selected.folds} блоках", f"{survived} из {select}")
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
            f"Настоящих признаков здесь нет вовсе, {reached} до верха — тоже по случайности. "
            "Высокий столбик — повод присмотреться к признаку, но не доказательство."
        )
    else:
        st.caption(
            f"Настоящий сигнал лежит в столбцах #0–#{informative - 1} (на графике они тёмные), и отбор нашёл "
            f"{found} из {informative}. Во всех {selected.folds} блоках отобраны снова {survived} из {select}: "
            "настоящие признаки держатся устойчиво, но и шумовой может случайно дотянуться до верха, "
            "так что по одной высоте столбика их не отличить."
        )

    # Two training parts of a k-fold split share k - 2 of their k - 1 blocks.
    shared = round(100 * (selected.folds - 2) / (selected.folds - 1))
    st.caption(
        f"Для сравнения: если бы два отбора брали по {select} столбцов из {settings.features} наугад, общих у "
        f"них было бы в среднем {chance:.2f}. Отборы на двух разных блоках здесь делят "
        f"{selected.mean_pairwise_overlap:.1f} общих признака — больше, потому что обучающие части двух блоков "
        f"на {shared} % состоят из одних и тех же строк и одни и те же случайные совпадения отбор находит в обеих"
        + (", а настоящие признаки выбираются почти каждый раз." if settings.signal else ".")
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
        st.caption(
            "F — насколько сильно среднее признака различается между классами по сравнению с разбросом "
            "внутри классов: чем больше, тем «значимее» признак выглядит. p — вероятность увидеть такое "
            "различие случайно, если связи на самом деле нет. Последний столбец — высота столбика на графике."
        )


def _pvalues_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("Почему отбор нашёл «значимое»")
    st.plotly_chart(pvalues_figure(np.asarray(_p_values(settings))), use_container_width=True)

    expected = expected_false_positives(selected.tested)
    left, right = st.columns(2)
    left.metric(
        "Признаков с p < 0.05",
        f"{selected.significant_at_05} из {selected.tested}",
        help="Порог 0.05 означает «такая связь возникает случайно реже чем в одном случае "
        "из двадцати». Проверьте две тысячи случайных столбцов — и сотня порог пройдёт.",
    )
    right.metric(
        "Выдержали поправку Бонферрони",
        str(selected.significant_after_bonferroni),
        help="Поправка делит порог на число проверок. Случайные кандидаты её не проходят — "
        "а отбор признаков о ней и не спрашивает.",
    )
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

    quick_start(
        st,
        purpose=(
            "Утечка данных — модель при обучении видит то, по чему её потом проверяют. Здесь в данных только "
            "шум, и честный ответ — 0.5."
        ),
        steps=(
            "Сравните два числа наверху: **«С утечкой»** — 0.757, **«Честно»** — 0.510.",
            "Меняйте **«Отбираем признаков»** и **«Блоков кросс-валидации»**: разрыв не исчезает.",
            "Поднимите **«Сила настоящего сигнала»**: честная оценка вырастет — на этот раз законно.",
        ),
        notice="отбор признаков, сделанный до разбиения выборки, находит закономерность даже в чистом шуме.",
    )
    how_to_read(
        st,
        what=(
            "Утечка данных — это когда модель при обучении так или иначе видит то, по чему её "
            "потом проверяют. Оценка выходит завышенной, причём молча: ничего не падает, "
            "никакой ошибки в коде нет. Здесь взят самый частый её вид — отбор признаков, "
            "сделанный один раз на всей таблице, до разбиения на обучающую и отложенную части. "
            "Данные нарочно взяты пустые: триста строк, две тысячи столбцов случайного шума, "
            "предсказывать в них нечего. Честная оценка даёт около 0.500 — как подброшенная "
            "монета, и это правильный ответ. Оценка с утечкой — около 0.760."
        ),
        controls=(
            "**Данные** — размер таблицы и сила настоящего сигнала; при силе 0 в данных нет "
            "ничего, что можно предсказать. **Отбор признаков** — сколько столбцов оставить "
            "модели; это и есть тот самый шаг, который совершает утечку. **Проверка** — на "
            "сколько блоков делить выборку."
        ),
        watch=(
            "Два первых числа наверху: «с утечкой» и «честно». Второе должно держаться около "
            "0.500 при любых настройках — потому что предсказывать нечего. Первое заметно выше, "
            "и разрыв между ними не исчезает ни от смены зерна, ни от числа блоков. Поднимите "
            "«силу настоящего сигнала» — честная оценка законно вырастет, и дальше по одному "
            "числу вы уже не отличите, где настоящий сигнал, а где утечка. В этом и трудность: "
            "на реальных данных вы видите только одно число."
        ),
        data=(
            "Таблица генерируется на месте из нормального шума, метки классов — тоже случайные "
            "и сбалансированные. Правильный ответ известен заранее: 0.500. Это и делает "
            "страницу доказательством, а не иллюстрацией. Сигнал, если его включить, — сдвиг "
            f"среднего у класса 1 в первых {INFORMATIVE} столбцах."
        ),
    )
    reference(
        st,
        "leakage",
        "feature_selection",
        "cross_validation",
        "accuracy",
        "baseline",
        "p_value",
        "multiple_testing",
    )

    settings, select, folds = _controls()
    result = _scores(settings, select, folds)
    selected = _selection(settings, select, folds)

    _data_section(settings, select, folds)
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
