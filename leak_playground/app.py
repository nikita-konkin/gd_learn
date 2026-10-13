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
SELECT_LABEL = "Сколько признаков отбирать"
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
    st.sidebar.caption("Эти настройки меняют только сами данные.")
    count = st.sidebar.slider(
        OBSERVATIONS_LABEL, 100, 600, DEFAULT.observations, step=50,
        help="Число строк таблицы. Чем их меньше относительно числа столбцов, "
        "тем легче найти в шуме мнимую закономерность.",
    )
    width = st.sidebar.slider(
        FEATURES_LABEL, 200, 4000, DEFAULT.features, step=200,
        help="Число столбцов-признаков. Если сигнал выключен, все они — чистый шум: "
        "случайные числа, никак не связанные с предсказываемым классом.",
    )
    signal = st.sidebar.slider(
        SIGNAL_LABEL, 0.0, 1.0, DEFAULT.signal, step=0.05,
        help=f"На сколько в среднем значения в столбцах #0–#{INFORMATIVE - 1} у объектов класса 1 больше, "
        "чем у класса 0. При нуле признаки с классом не связаны, предсказывать нечего, и любая оценка "
        "выше 0.500 мнимая. При 1 сдвиг равен стандартному отклонению шума, и по этим столбцам классы "
        "хорошо различимы.",
    )
    seed = st.sidebar.slider(
        SEED_LABEL, 0, 20, DEFAULT.seed,
        help="Другое зерно даёт другой шум при тех же настройках. Разрыв при этом не исчезает: "
        "это не случайность одного набора данных.",
    )

    st.sidebar.header("Отбор признаков")
    st.sidebar.caption("Эта настройка меняет только отбор.")
    select = st.sidebar.slider(
        SELECT_LABEL, 5, 100, DEFAULT_SELECT, step=5,
        help=(
            "Сколько самых «значимых» столбцов оставить модели. Именно этот шаг приводит к утечке, если "
            "выполнен до разбиения выборки."
        ),
    )

    st.sidebar.header("Проверка")
    st.sidebar.caption("Эта настройка меняет только разбиение на блоки.")
    folds = st.sidebar.slider(
        FOLDS_LABEL, 3, 10, DEFAULT_FOLDS,
        help=(
            "На сколько блоков делится выборка при проверке: каждый по очереди становится отложенным, "
            "остальные — обучающими."
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
        rest = settings.features - informative
        signal = (
            f"Сейчас его сила {settings.signal:.2f}: у объектов класса 1 значения в столбцах {columns} в среднем "
            f"на {settings.signal:.2f} больше, чем у класса 0, и по этим {informative} столбцам классы "
            f"действительно различимы. Остальные {rest} {plural(rest, 'столбец', 'столбца', 'столбцов')} — "
            "по-прежнему шум."
        )
    else:
        signal = (
            "Сейчас сигнала нет: сила равна 0, признаки и классы независимы, предсказывать нечего. Если сдвинуть "
            f"ползунок «{SIGNAL_LABEL}», например, до 0.6, значения в столбцах {columns} у объектов класса 1 "
            "станут в среднем на 0.6 больше, чем у класса 0, и по этим столбцам классы можно будет различить."
        )

    st.subheader("Данные и задача")
    st.markdown(
        f"- **Таблица** — {observations(settings.observations)} (строки) и {features(settings.features)} "
        f"(столбцы #0 … #{settings.features - 1}). Данные синтетические: каждое значение — случайное число "
        "из нормального распределения со средним 0 и стандартным отклонением 1, а сама таблица генерируется "
        "прямо на странице. Поэтому правильный ответ известен заранее.\n"
        "- **Цель** — класс каждой строки, 0 или 1. Классы тоже назначаются случайно, примерно поровну. "
        "Модель — логистическая регрессия, качество измеряется долей правильных ответов на кросс-валидации; "
        f"при угадывании наугад она равна {as_printed(TRUTH)}.\n"
        f"- **Сигнал** — настоящая связь признаков с классом, которую модели и следовало бы найти. {signal}\n"
        f"- **Утечка** — ошибка в порядке шагов. Перед обучением из {features(settings.features)} оставляют "
        f"{select}, сильнее всего различающихся между классами. Если отобрать их один раз по всей таблице, "
        "отбор учтёт и строки, на которых модель затем проверяют, и найдёт «различия» даже в чистом шуме. "
        f"Честная оценка требует повторять отбор внутри каждого из {folds} блоков кросс-валидации, только "
        "по обучающей части."
    )

    matrix, labels = _data(settings)
    shown = min(PREVIEW_COLUMNS, settings.features)
    preview = pd.DataFrame(matrix[:PREVIEW_ROWS, :shown].round(2), columns=[f"#{index}" for index in range(shown)])
    preview["класс"] = labels[:PREVIEW_ROWS]
    means = [float(matrix[labels == label, :informative].mean()) for label in (0, 1)]
    with st.expander("Как выглядит таблица"):
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.caption(
            f"Показаны первые {PREVIEW_ROWS} строк и {shown} столбцов из {settings.features}. Среднее по "
            f"столбцам {columns} у класса 0 равно {means[0]:.2f}, у класса 1 — {means[1]:.2f}. Разница этих "
            "средних и есть сигнал: при силе 0 они почти совпадают, при силе s различаются примерно на s."
        )


def _headline(result: Scores, settings: NoiseSettings, select: int, folds: int) -> None:
    st.subheader("Две оценки одной модели")
    left, middle, right = st.columns(3)
    left.metric(
        "С утечкой",
        as_printed(result.leaky),
        help="Доля правильных ответов, когда отбор признаков сделан один раз по всей таблице, "
        "т. е. с учётом отложенных строк.",
    )
    middle.metric(
        "Честно",
        as_printed(result.honest),
        help="Та же модель, но отбор признаков повторяется внутри каждого блока, "
        "только по обучающей части. Таков правильный порядок.",
    )
    right.metric(
        "Разрыв",
        f"+{as_printed(result.gap)}",
        help="Разница между двумя оценками: на столько вы введёте в заблуждение себя и заказчика, "
        "если выполните отбор признаков не на том шаге.",
    )

    if settings.signal == 0:
        st.caption(
            f"Истинное значение — {as_printed(TRUTH)}: классы сбалансированы, а признаки никак не связаны "
            f"с целью. Разрыв в {as_printed(result.gap)} создан отбором, который видел отложенные строки."
        )
    else:
        st.caption(
            f"Сигнал добавлен в {features(min(INFORMATIVE, settings.features))}, его сила {settings.signal:.2f}. "
            "Теперь честная оценка выше 0.500 обоснованно, а утечка по-прежнему добавляет к ней свою "
            "прибавку, и по одному числу эти две прибавки не различить."
        )


def _recurrence_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("Что именно отобралось")
    informative = min(INFORMATIVE, settings.features)
    st.markdown(
        f"Отбор по всей таблице выбрал {select} столбцов; их номера подписаны под графиком. Чтобы проверить, "
        f"насколько надёжен этот выбор, повторим отбор {selected.folds} раз, как это делает честная ветвь, — "
        "каждый раз только по обучающей части одного блока. Высота столбика показывает, в скольких блоках "
        "признак отобран снова.\n\n"
        "Признак, действительно связанный с классом, отбирается почти при любом разбиении, и его столбик "
        "доходит до пунктира. Признак, попавший в выбор из-за случайного совпадения, на других строках уже "
        "не выделяется, и его столбик остаётся низким."
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
            reached = f"и всё же {survived} из {select} {plural(survived, 'дошёл', 'дошли', 'дошли')} до пунктира"
        else:
            reached = f"и ни один из {select} не дошёл до пунктира"
        st.caption(
            f"Настоящих признаков здесь нет, {reached} — тоже случайно. "
            "Высокий столбик — повод проверить признак, но не доказательство его связи с классом."
        )
    else:
        st.caption(
            f"Настоящий сигнал заложен в столбцы #0–#{informative - 1} (на графике они тёмные), и отбор нашёл "
            f"{found} из {informative}. Во всех {selected.folds} блоках снова отобраны {survived} из {select}: "
            "настоящие признаки отбираются устойчиво, но и шумовой может случайно дойти до пунктира, "
            "поэтому по одной высоте столбика их не различить."
        )

    # Two training parts of a k-fold split share k - 2 of their k - 1 blocks.
    shared = round(100 * (selected.folds - 2) / (selected.folds - 1))
    st.caption(
        f"Для сравнения: два отбора по {select} столбцов из {settings.features}, сделанные наугад, имели бы "
        f"в среднем {chance:.2f} общих признака. Отборы на двух разных блоках здесь имеют в среднем "
        f"{selected.mean_pairwise_overlap:.1f} общих признака. Их больше, потому что обучающие части двух "
        f"блоков на {shared} % состоят из одних и тех же строк и отбор находит в них одни и те же случайные "
        "совпадения" + (", а настоящие признаки отбираются почти каждый раз." if settings.signal else ".")
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
            "F показывает, насколько сильно среднее признака различается между классами по сравнению "
            "с разбросом внутри классов: чем больше F, тем «значимее» выглядит признак. p — вероятность "
            "получить различие не меньше наблюдаемого, если связи на самом деле нет. Последний столбец "
            "совпадает с высотой столбика на графике."
        )


def _pvalues_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("Почему отбор нашёл «значимое»")
    st.plotly_chart(pvalues_figure(np.asarray(_p_values(settings))), use_container_width=True)

    expected = expected_false_positives(selected.tested)
    left, right = st.columns(2)
    left.metric(
        "Признаков с p < 0.05",
        f"{selected.significant_at_05} из {selected.tested}",
        help="Порог 0.05 означает, что такая связь возникает случайно реже чем в одном случае "
        "из двадцати. Из двух тысяч случайных столбцов этот порог пройдёт около сотни.",
    )
    right.metric(
        "Выдержали поправку Бонферрони",
        str(selected.significant_after_bonferroni),
        help="Поправка делит порог на число проверок, и случайные кандидаты её не проходят. "
        "Но отбор признаков эту поправку не учитывает.",
    )
    st.caption(
        f"Проверено {features(selected.tested)}; при пороге 0.05 около {expected:.0f} из них пройдут его "
        f"случайно. Прошли {selected.significant_at_05}. "
        f"Поправку на {selected.tested} проверок выдержали "
        f"{selected.significant_after_bonferroni}, и всё же отбор взял {select} лучших "
        "и передал их модели."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Отбор признаков до разбиения выборки — самая частая и самая незаметная ошибка. "
        "Здесь её последствия измерены на данных, в которых предсказывать нечего."
    )

    quick_start(
        st,
        purpose=(
            "Утечка данных возникает, когда модель при обучении видит то, по чему её затем проверяют. "
            "Здесь данные — чистый шум, и честный результат — 0.5."
        ),
        steps=(
            "Сравните два числа наверху: **«С утечкой»** — 0.757, **«Честно»** — 0.510.",
            "Меняйте **«Сколько признаков отбирать»** и **«Блоков кросс-валидации»**: разрыв не исчезает.",
            "Сдвиньте вправо **«Сила настоящего сигнала»**: честная оценка вырастет, на этот раз обоснованно.",
        ),
        notice="отбор признаков, сделанный до разбиения выборки, находит закономерность даже в чистом шуме.",
    )
    how_to_read(
        st,
        what=(
            "Утечка данных возникает, когда модель при обучении так или иначе видит то, по чему её "
            "затем проверяют. Оценка оказывается завышенной, причём незаметно: программа не "
            "завершается с ошибкой, и формально код верен. Здесь показан самый частый вид утечки — "
            "отбор признаков, сделанный один раз по всей таблице до разбиения на обучающую и "
            "отложенную части. Данные намеренно не содержат закономерностей: триста строк и две "
            "тысячи столбцов случайного шума. Честная оценка близка к 0.500, как у подброшенной "
            "монеты, и это правильный ответ. Оценка с утечкой — около 0.760."
        ),
        controls=(
            "**Данные** — размер таблицы и сила настоящего сигнала; при силе 0 в данных нет "
            "ничего, что можно предсказать. **Отбор признаков** — сколько столбцов оставить "
            "модели; именно этот шаг и приводит к утечке. **Проверка** — на сколько блоков "
            "делить выборку."
        ),
        watch=(
            "Смотрите на два первых числа: «С утечкой» и «Честно». Пока сигнал выключен, второе "
            "при любых настройках остаётся около 0.500, потому что предсказывать нечего. Первое "
            "заметно выше, и разрыв между ними не исчезает ни при смене зерна, ни при изменении "
            "числа блоков. Если увеличить силу настоящего сигнала, честная оценка обоснованно "
            "вырастет, и по одному числу уже не отличить настоящий сигнал от утечки. В этом и "
            "трудность: на реальных данных видно только одно число."
        ),
        data=(
            "Таблица генерируется прямо на странице из нормального шума; метки классов тоже "
            "случайны и сбалансированы. Правильный ответ известен заранее — 0.500, и это делает "
            "страницу доказательством, а не иллюстрацией. Включённый сигнал сдвигает среднее "
            f"у класса 1 в первых {INFORMATIVE} столбцах."
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
        "Разрыв сохраняется в каждом блоке, а не возникает в среднем из-за удачного разбиения. "
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
            "- Не меняет модель: везде используется логистическая регрессия. Утечка не зависит "
            "от выбора модели — это свойство порядка операций.\n"
            "- Не выполняет вложенную кросс-валидацию: подбор гиперпараметров на той же "
            "проверочной части — отдельная тема.\n"
            f"- Не моделирует реальный сигнал: включённый сигнал всегда добавляется в первые "
            f"{INFORMATIVE} столбцов как сдвиг среднего. Это лишь способ показать, что по одному "
            "числу утечку от сигнала не отличить."
        )


if __name__ == "__main__":
    main()
