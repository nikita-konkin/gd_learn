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
    block_sizes,
    chance_overlap,
    expected_false_positives,
    scores,
    selection,
)
from leak_playground.noise import INFORMATIVE, NoiseSettings, make_data  # noqa: E402
from leak_playground.plotting import pvalues_figure, recurrence_figure, scores_figure, split_figure  # noqa: E402
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


def _data_section(settings: NoiseSettings, select: int) -> None:
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

    st.subheader("1. Данные и задача")
    st.markdown(
        f"- **Таблица** — {observations(settings.observations)} (строки) и {features(settings.features)} "
        f"(столбцы #0 … #{settings.features - 1}). Данные синтетические: каждое значение — случайное число "
        "из нормального распределения со средним 0 и стандартным отклонением 1, а сама таблица генерируется "
        "прямо на странице. Поэтому правильный ответ известен заранее.\n"
        "- **Цель** — класс каждой строки, 0 или 1. Классы тоже назначаются случайно, примерно поровну. "
        "Модель — логистическая регрессия, качество измеряется долей правильных ответов на кросс-валидации; "
        f"при угадывании наугад она равна {as_printed(TRUTH)}.\n"
        f"- **Сигнал** — настоящая связь признаков с классом, которую модели и следовало бы найти. {signal}\n"
        f"- **Отбор признаков** — перед обучением из {features(settings.features)} оставляют {select}, "
        "сильнее всего различающихся между классами: модели проще работать с немногими столбцами, "
        "чем с тысячами. Утечка возникает, если этот шаг сделан не в том месте; разделы 2 и 3 показывают, "
        "в каком."
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


def _split_section(settings: NoiseSettings, folds: int) -> None:
    """Cross-validation itself, drawn before any score that is counted in its blocks."""
    _, labels = _data(settings)
    sizes = block_sizes(labels, folds)
    low, high = min(sizes), max(sizes)
    size = f"{low}" if low == high else f"{low}–{high}"
    st.subheader("2. Как проверяют модель: кросс-валидация")
    st.markdown(
        f"Проверять модель на тех же строках, на которых она обучалась, бессмысленно: так она «узнаёт» "
        f"ответы, а не предсказывает их. Поэтому {observations(settings.observations)} делят на {folds} "
        f"блоков по {size} {plural(high, 'строке', 'строки', 'строк')} и обучают модель {folds} раз. Каждый "
        "раз один блок откладывают для проверки, а по остальным обучают. Получается "
        f"{folds} {plural(folds, 'оценка', 'оценки', 'оценок')}, по одной на блок; итоговая оценка — их среднее."
    )
    st.plotly_chart(split_figure(sizes), use_container_width=True)
    st.caption(
        "Строка схемы — одно обучение, столбец — один блок строк. Тёмная клетка — блок, отложенный "
        "для проверки: модель его не видит, пока не обучится; по светлым клеткам модель обучают. "
        "Число блоков задаёт настройка "
        f"«{FOLDS_LABEL}»."
    )


def _pipelines_section(settings: NoiseSettings, select: int, folds: int) -> None:
    """The two orders of steps side by side; the only difference is where the selection sits."""
    st.subheader("3. Где возникает утечка: два порядка шагов")
    st.markdown(
        "Кросс-валидация честна, только если всё, что модель узнаёт о данных, она узнаёт по обучающим "
        "блокам. Отбор признаков тоже чему-то учится по данным: он смотрит, какие столбцы различают классы. "
        "Значит, и его нужно делать внутри каждого обучения."
    )
    left, right = st.columns(2)
    left.markdown(
        ":orange[**С утечкой**]\n\n"
        f"1. Отобрать {select} признаков по всем {settings.observations} строкам, включая те, на которых "
        "модель потом проверят.\n"
        f"2. Для каждого из {folds} блоков: обучить модель на остальных блоках и проверить на этом.\n\n"
        "Отбор заранее «видел» проверочные блоки и выбрал столбцы, которые случайно различают классы "
        "именно на них."
    )
    right.markdown(
        ":blue[**Честно**]\n\n"
        f"1. Для каждого из {folds} блоков:\n"
        f"    - отобрать {select} признаков только по остальным блокам;\n"
        "    - обучить модель на тех же блоках;\n"
        "    - проверить на отложенном блоке.\n\n"
        "Отобранные столбцы в каждом обучении свои, а проверочный блок не участвует ни в отборе, "
        "ни в обучении."
    )
    st.caption(
        "Обе ветви масштабируют признаки и обучают одну и ту же модель на одних и тех же блоках; различаются "
        "они только тем, где сделан отбор."
    )


def _headline(result: Scores, settings: NoiseSettings, select: int, folds: int) -> None:
    st.subheader("4. Результат кросс-валидации")
    st.markdown(
        f"Обе ветви прошли одни и те же {folds} {plural(folds, 'обучение', 'обучения', 'обучений')}. Ниже — "
        f"их итоговые оценки, т. е. средняя доля правильных ответов по {folds} проверочным блокам."
    )
    left, middle, right = st.columns(3)
    left.metric(
        "С утечкой",
        as_printed(result.leaky),
        help="Средняя доля правильных ответов на проверочных блоках, когда отбор признаков сделан "
        "один раз по всей таблице, т. е. с учётом проверочных строк.",
    )
    middle.metric(
        "Честно",
        as_printed(result.honest),
        help="Та же модель на тех же блоках, но отбор признаков повторяется в каждом обучении "
        "и только по обучающим блокам. Таков правильный порядок.",
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
            f"с целью. Разрыв в {as_printed(result.gap)} создан отбором, который видел проверочные строки."
        )
    else:
        st.caption(
            f"Сигнал добавлен в {features(min(INFORMATIVE, settings.features))}, его сила {settings.signal:.2f}. "
            "Теперь честная оценка выше 0.500 обоснованно, а утечка по-прежнему добавляет к ней свою "
            "прибавку, и по одному числу эти две прибавки не различить."
        )
    st.caption(
        f"{observations(settings.observations)}, {features(settings.features)}, отобрано {select}, блоков {folds}."
    )

    st.markdown(
        "Среднее легко списать на одно удачное разбиение, поэтому на диаграмме те же оценки показаны "
        "по отдельности: пара столбцов — один проверочный блок, т. е. одна строка схемы из раздела 2."
    )
    st.plotly_chart(scores_figure(result.leaky_folds, result.honest_folds), use_container_width=True)
    st.caption(
        "Высота столбца — доля правильных ответов модели на строках этого блока. Оранжевый — ветвь "
        "с утечкой, синий — честная, пунктир — истинное значение. Числа над диаграммой — средние по "
        "столбцам своего цвета. Разрыв сохраняется в каждом блоке, а не возникает в среднем из-за "
        "удачного разбиения."
    )


def _recurrence_section(selected: Selection, settings: NoiseSettings, select: int) -> None:
    st.subheader("5. Откуда разрыв: что именно отобралось")
    informative = min(INFORMATIVE, settings.features)
    st.markdown(
        f"Ветвь с утечкой работает с {select} признаками, отобранными по всей таблице. Связаны ли они "
        "с классом на самом деле? Настоящий признак выделяется на любой части данных, поэтому отбор, "
        "повторённый без части строк, выберет его снова. Честная ветвь как раз повторяет отбор "
        f"{selected.folds} раз — в каждом обучении, без проверочного блока. Диаграмма показывает для каждого "
        f"из {select} признаков ветви с утечкой, сколько из этих {selected.folds} повторных отборов выбрали "
        "его снова.\n\n"
        "Настоящий признак доходит до пунктира почти при любом разбиении. Признак, попавший в отбор из-за "
        "случайного совпадения, без части строк перестаёт выделяться, и его столбец остаётся низким. "
        "Такие признаки «различают» классы лишь на тех строках, по которым их отобрали, в том числе на "
        "проверочных. Поэтому модель с утечкой угадывает проверочные строки лучше, чем позволяют данные."
    )
    st.plotly_chart(recurrence_figure(selected, informative if settings.signal else 0), use_container_width=True)

    chance = chance_overlap(select, settings.features)
    survived = sum(1 for value in selected.recurrence if value == selected.folds)
    found = sum(1 for index in selected.indices if index < informative)

    left, right = st.columns(2)
    left.metric(f"Выбраны снова во всех {selected.folds} повторных отборах", f"{survived} из {select}")
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
            f"{found} из {informative}. Во всех {selected.folds} повторных отборах снова выбраны {survived} "
            f"из {select}: "
            "настоящие признаки отбираются устойчиво, но и шумовой может случайно дойти до пунктира, "
            "поэтому по одной высоте столбика их не различить."
        )

    # Two training parts of a k-fold split share k - 2 of their k - 1 blocks.
    shared = round(100 * (selected.folds - 2) / (selected.folds - 1))
    st.caption(
        f"Для сравнения: два отбора по {select} столбцов из {settings.features}, сделанные наугад, имели бы "
        f"в среднем {chance:.2f} общих признака. Два повторных отбора здесь имеют в среднем "
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
    st.subheader("6. Почему шум выглядит значимым")
    expected = expected_false_positives(selected.tested)
    st.markdown(
        "Отбор оценивает каждый столбец F-тестом: насколько различаются средние двух классов. Результат "
        "теста — p-значение, т. е. вероятность получить такое различие случайно, если связи нет. Отбор "
        f"берёт {select} столбцов с наименьшим p.\n\n"
        f"Гистограмма показывает p-значения всех {features(selected.tested)}, сгруппированные по интервалам "
        "шириной 0.05. Без сигнала они распределены равномерно: в каждый интервал попадает около "
        f"{expected:.0f} столбцов. Оранжевый столбец — признаки с p < 0.05. Их около {expected:.0f}, "
        f"{'почти все' if settings.signal else 'все'} они случайны, и именно из них отбор берёт свои {select}."
    )
    st.plotly_chart(pvalues_figure(np.asarray(_p_values(settings))), use_container_width=True)

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
            "Прочитайте разделы 2 и 3: как модель проверяют и где в этой проверке утечка.",
            "В разделе 4 сравните **«С утечкой»** — 0.757 и **«Честно»** — 0.510.",
            "Меняйте **«Сколько признаков отбирать»** и **«Блоков кросс-валидации»**: разрыв не исчезает.",
            "Сдвиньте вправо **«Сила настоящего сигнала»**: честная оценка обоснованно вырастет.",
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
            "Разделы идут по порядку: данные, устройство кросс-валидации, два порядка шагов, результат "
            "и две причины разрыва. Главные числа — «С утечкой» и «Честно» в разделе 4. Пока сигнал выключен, второе "
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

    _data_section(settings, select)
    _split_section(settings, folds)
    _pipelines_section(settings, select, folds)
    _headline(result, settings, select, folds)

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
