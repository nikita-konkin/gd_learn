"""Streamlit interface for the data-preparation playground.

Two lessons of data preparation, each given handles. Cleaning: the order of
steps and the choice of statistic, on six rows with planted defects and on a
thousand. Speed and memory: who runs the loop, and what a Python int really
costs — measured live, in the browser.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from data_playground.cleaning import (
    AFTER,
    MEAN,
    MEDIAN,
    ORDERS,
    OUTLIER_LIMIT,
    STATISTICS,
    Cleaned,
    Estimate,
    Readings,
    ReadingsSettings,
    clean_defect_table,
    clean_signal,
    defect_table,
    estimate,
    inspect,
    make_readings,
)
from data_playground.plotting import cleaned_histogram, timing_figure
from data_playground.speed import DOT, SUM_SIZE, Memory, Timing, memory, results_agree, sum_of_squares
from playground_common.explain import how_to_read, prose, quick_start, reference, use_terms
from playground_common.links import other_playgrounds
from playground_common.wording import duration, plural

TITLE = "Конвейер подготовки данных"

ORDER_LABEL = "Когда считать статистику для заполнения"
STATISTIC_LABEL = "Чем заполнять пропуски"
ROWS_LABEL = "Строк"
MISSING_LABEL = "Пропусков, %"
OUTLIERS_LABEL = "Невозможных значений, %"
SEED_LABEL = "Зерно генерации"
SIZE_LABEL = "Сколько чисел возводить в квадрат"
REMEASURE_LABEL = "Измерить ещё раз"

SIZES = [10_000, 100_000, 1_000_000]
DEFAULT_READINGS = ReadingsSettings()


@lru_cache(maxsize=32)
def _readings(settings: ReadingsSettings) -> Readings:
    return make_readings(settings)


@lru_cache(maxsize=64)
def _estimate(settings: ReadingsSettings, statistic: str, order: str) -> Estimate:
    return estimate(_readings(settings), statistic, order)


@lru_cache(maxsize=8)
def _timings(size: int) -> tuple[Timing, ...]:
    return sum_of_squares(size)


@lru_cache(maxsize=1)
def _memory() -> Memory:
    return memory()


def _controls() -> tuple[str, str, ReadingsSettings, int]:
    """The sidebar: how to clean, what to clean, how much to time — three separate groups."""
    st.sidebar.header("Очистка")
    st.sidebar.caption("Эти настройки меняют только порядок и способ очистки.")
    order = st.sidebar.radio(
        ORDER_LABEL,
        list(ORDERS),
        index=0,
        help=(
            "Главная настройка страницы. Пропуски нужно чем-то заполнить, и это значение "
            "вычисляется по самому столбцу. Если вычислить его до удаления невозможных "
            "значений, они войдут в расчёт, и очистка впишет в таблицу новую ошибку."
        ),
    )
    statistic = st.sidebar.radio(
        STATISTIC_LABEL,
        list(STATISTICS),
        index=0,
        help=(
            "Среднее чувствительно к выбросам: одно значение −900 заметно смещает его. "
            "Медиана — значение в середине упорядоченного ряда, и выброс сдвигает её лишь на одну позицию."
        ),
    )

    st.sidebar.header("Тысяча строк")
    st.sidebar.caption("Эти настройки меняют только сами данные второй вкладки.")
    rows = st.sidebar.slider(
        ROWS_LABEL, 200, 5000, DEFAULT_READINGS.rows, step=200,
        help="Сколько показаний сгенерировать для второй вкладки.",
    )
    missing = st.sidebar.slider(
        MISSING_LABEL, 0, 30, round(DEFAULT_READINGS.missing * 100),
        help="Какую долю показаний «потерял» прибор; эти ячейки придётся заполнять.",
    )
    outliers = st.sidebar.slider(
        OUTLIERS_LABEL, 0, 15, round(DEFAULT_READINGS.outliers * 100),
        help=(
            "Какую долю показаний прибор вернул как −900 дБм. Это не слабый сигнал, "
            "а код ошибки: уровень ниже −120 дБм физически невозможен."
        ),
    )
    seed = st.sidebar.slider(
        SEED_LABEL, 0, 20, DEFAULT_READINGS.seed,
        help="Задаёт начальное состояние генератора: при другом зерне получаются другие данные с теми же настройками.",
    )

    st.sidebar.header("Скорость")
    st.sidebar.caption("Эта настройка меняет только объём замера.")
    size = st.sidebar.select_slider(
        SIZE_LABEL, SIZES, value=SUM_SIZE,
        help="Чем больше чисел, тем заметнее разница между циклом на Python и NumPy.",
    )

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/data"))

    settings = ReadingsSettings(rows=rows, missing=missing / 100, outliers=outliers / 100, seed=seed)
    return order, statistic, settings, size


def _cleaned_view(cleaned: Cleaned) -> pd.DataFrame:
    view = cleaned.table.copy()
    view["сигнал"] = view["сигнал"].round(2)
    view["заполнено"] = ["← заполнено" if flag else "" for flag in cleaned.filled]
    return view


def _steps(order: str, statistic: str) -> str:
    """The cleaning chain as the reader should see it, in the order the chosen setting runs it."""
    noun = {MEDIAN: "медиану", MEAN: "среднее"}[statistic]
    limit = f"{OUTLIER_LIMIT:.0f}".replace("-", "−")
    replace = f"заменить невозможные значения (ниже {limit} дБм) пропусками"
    compute = f"вычислить {noun} столбца «сигнал»"
    middle = (replace, compute) if order == AFTER else (compute, replace)
    chain = (
        "удалить полные дубликаты",
        "привести статус к нижнему регистру",
        "удалить строки без узла",
        *middle,
        f"вписать {'её' if statistic == MEDIAN else 'его'} во все пустые ячейки столбца «сигнал»",
    )
    return "\n".join(f"{index}. {step[0].upper()}{step[1:]}." for index, step in enumerate(chain, start=1))


def _defect_tab(order: str, statistic: str) -> None:
    raw = defect_table()
    found = inspect(raw)
    st.subheader("1. Исходная таблица")
    prose(st,
        "Шесть показаний узлов связи. **узел** — имя приёмника, **сигнал** — уровень принятого сигнала "
        "в дБм (децибелах относительно 1 мВт; чем ближе к нулю, тем сильнее сигнал, реальные значения лежат "
        "примерно от −30 до −120), **статус** — исправен ли приёмник. В таблицу намеренно внесены четыре "
        "дефекта: пропуски, дубликат, разное написание статуса и физически невозможное значение −900."
    )
    st.dataframe(raw, use_container_width=True, hide_index=True)

    st.subheader("2. Что нашла проверка")

    left, middle, right = st.columns(3)
    left.metric("Пропусков", str(sum(found.missing.values())))
    middle.metric(
        "Полных дубликатов",
        str(found.duplicates),
        help="Строки, совпадающие во всех столбцах. Одно показание, записанное дважды, "
        "смещает любую статистику в свою сторону.",
    )
    right.metric(
        "Вариантов статуса",
        str(len(found.statuses)),
        help="Сколько разных написаний встретилось в столбце статуса. Для человека «ok» и "
        "«OK» — одно и то же, для программы — две разные категории.",
    )
    st.caption(
        f"Пропуски по столбцам: {', '.join(f'{column} — {count}' for column, count in found.missing.items())}. "
        f"Статусы: {', '.join(found.statuses)} — «ok» и «OK» означают одно и то же."
    )

    cleaned = clean_defect_table(statistic, order)
    st.subheader("3. Очистка")
    prose(st,
        "Очистка — цепочка шагов. При выбранных настройках они выполняются в таком порядке:\n\n"
        + _steps(order, statistic)
        + "\n\nПорядок шагов 4 и 5 задаёт настройка «"
        + ORDER_LABEL
        + "», а статистику — «"
        + STATISTIC_LABEL
        + "». Результат — таблица ниже; заполненные ячейки отмечены справа."
    )
    st.dataframe(_cleaned_view(cleaned), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    left.metric(
        "Заполнено значением",
        f"{cleaned.fill_value:.2f}",
        help="Число, которое очистка вписала во все пустые ячейки столбца «сигнал».",
    )
    right.metric(
        "Невозможных значений после очистки",
        str(cleaned.impossible_left),
        help=f"Сколько осталось показаний ниже физического предела {OUTLIER_LIMIT:.0f} дБм. "
        "Должно быть ноль, однако очистка может вписать их сама.",
    )

    if (order, statistic) == (AFTER, MEDIAN):
        st.info(
            "Порядок правильный: выбросы сначала превращены в пропуски, затем все пропуски заполнены "
            f"медианой. Пропусков осталось: {cleaned.missing_left}, в обеих заполненных ячейках — "
            f"{cleaned.fill_value:.2f}, в пределах физически возможного."
        )
    if cleaned.impossible_left:
        noun = plural(cleaned.impossible_left, "невозможное значение", "невозможных значения", "невозможных значений")
        st.error(
            f"Очистка сама вставила в таблицу {cleaned.impossible_left} {noun}: "
            f"среднее посчитано вместе с −900 и равно {cleaned.fill_value:.2f}, что ниже физического предела "
            f"{OUTLIER_LIMIT:.0f}. Заполнение внесло в таблицу новую ошибку вместо того, чтобы исправить старую."
        )
    elif order != AFTER:
        st.warning(
            f"Медиана устойчивее: даже посчитанная вместе с выбросом, она дала {cleaned.fill_value:.2f} "
            "вместо −48.65. Ошибка есть, но она незаметна и потому опаснее."
        )


def _readings_tab(settings: ReadingsSettings, order: str, statistic: str) -> None:
    readings = _readings(settings)
    result = _estimate(settings, statistic, order)
    st.subheader("1. Данные")
    prose(st,
        f"Та же задача на {settings.rows} показаниях трёх узлов. Сначала генерируются истинные уровни "
        "сигнала около −50 дБм, затем часть из них заменяется пропусками, а часть — кодом ошибки −900. "
        f"Сейчас в данных {int(readings.missing_mask.sum())} пропусков и "
        f"{int(readings.outlier_mask.sum())} значений −900; их доли задают настройки слева. Истинные "
        "уровни известны, поэтому можно измерить, насколько очистка исказила данные. Очистка та же, что "
        "на первой вкладке, с теми же настройками."
    )

    st.subheader("2. Насколько очистка исказила среднее")

    left, middle, right = st.columns(3)
    left.metric(
        "Истинное среднее",
        f"{result.true_mean:.2f}",
        help="Среднее по исходным показаниям, до внесения пропусков и значений −900.",
    )
    middle.metric(
        "Среднее после очистки",
        f"{result.cleaned_mean:.2f}",
        help="Среднее по таблице, которую выдала очистка. Именно это число и ушло бы в отчёт.",
    )
    right.metric(
        "Ошибка очистки",
        f"{result.error:+.2f}",
        help=(
            "Разница между ними, в дБм. Ноль означает, что очистка не исказила данные. "
            "В реальной задаче такую ошибку никто бы не заметил: истинное "
            "среднее там неизвестно."
        ),
    )

    cleaned = clean_signal(readings.table, statistic, order)
    st.subheader("3. Распределение после очистки")
    prose(st,
        "Гистограмма: по горизонтали уровень сигнала после очистки, по вертикали число строк с таким "
        "уровнем. Синие столбцы — измеренные значения, оранжевый — заполненные пропуски. Вертикальные "
        "линии отмечают истинное среднее и физический предел."
    )
    st.plotly_chart(
        cleaned_histogram(cleaned.table["сигнал"].to_numpy(), np.asarray(cleaned.filled, dtype=bool), result.true_mean),
        use_container_width=True,
    )
    st.caption(
        f"Все {result.filled} пропусков заполнены одним значением {result.fill_value:.2f}. "
        "Они образуют оранжевый столбец. Если он стоит внутри распределения, заполнение ничего не исказило; "
        "если в стороне от него, искажение внесла сама очистка."
    )
    if result.impossible_left:
        st.error(
            f"Значение заполнения {result.fill_value:.2f} ниже физического предела {OUTLIER_LIMIT:.0f}: "
            f"очистка создала {result.impossible_left} невозможных показаний."
        )


def _speed_tab(size: int) -> None:
    if st.button(REMEASURE_LABEL, help="Замер будет выполнен повторно; числа немного изменятся, "
                 "это обычный разброс измерений."):
        _timings.cache_clear()
    timings = _timings(size)
    slowest = timings[0].seconds
    spaced = f"{size:,}".replace(",", " ")

    st.subheader("1. Задача")
    prose(st,
        f"Одна и та же сумма квадратов {spaced} чисел вычисляется тремя способами. Алгоритм у них один; "
        "различается лишь то, кто выполняет цикл: интерпретатор Python, перебирающий числа по одному, "
        "или скомпилированный код внутри NumPy."
    )
    st.code(
        "sum(x * x for x in numbers)   # генератор Python: цикл в интерпретаторе\n"
        "np.sum(array ** 2)            # NumPy поэлементно: массив квадратов, затем сумма\n"
        "array @ array                 # NumPy, скалярное произведение: всё за один проход",
        language="python",
    )
    st.subheader("2. Время одного вызова")
    prose(st,
        "Длина полосы — время одного вызова. Шкала логарифмическая: каждое деление в десять раз больше "
        "предыдущего, поэтому разница на одно деление — это разница в десять раз."
    )
    st.plotly_chart(timing_figure(timings), use_container_width=True)
    columns = st.columns(len(timings))
    for column, timing in zip(columns, timings, strict=True):
        # A clock that never moved leaves nothing to divide by: say nothing rather than «×inf».
        delta = None
        if timing is not timings[0] and timing.seconds:
            delta = f"×{slowest / timing.seconds:.0f} быстрее"
        column.metric(timing.method, duration(timing.seconds), delta)
    st.metric("Результаты совпадают", "да" if results_agree(timings) else "нет")
    st.caption(
        "Время измерено только что, в этом браузере. В обычном Python на вашем компьютере оно будет другим: "
        "здесь Python работает внутри WebAssembly. Порядок величин "
        f"сохраняется, как и причина, по которой «{DOT}» обгоняет поэлементный вариант: он не создаёт "
        "промежуточный массив квадратов. Время указано на один вызов и усреднено по серии вызовов: браузер "
        "округляет показания часов до 0.1 мс или грубее, и быстрый вызов, измеренный "
        "отдельно, показал бы ноль."
    )

    st.subheader("3. Память: список против массива")
    prose(st,
        "Кроме скорости, массив выигрывает и в памяти. Список Python хранит не сами числа, а указатели "
        "на отдельные объекты-числа; массив NumPy хранит числа подряд, по восемь байт на каждое."
    )
    used = _memory()
    left, middle, right = st.columns(3)
    left.metric(
        "Список, МБ",
        f"{used.list_bytes / 1024**2:.1f}",
        help="Обычный список Python: на каждое число — отдельный объект плюс указатель на него.",
    )
    middle.metric(
        "Массив int64, МБ",
        f"{used.array_bytes / 1024**2:.1f}",
        help="Массив NumPy: один непрерывный участок памяти, ровно восемь байт на число, "
        "без объектов и указателей.",
    )
    right.metric("Разница, раз", f"{used.ratio:.1f}")
    pointer = f"{used.pointer_bytes} {plural(used.pointer_bytes, 'байт', 'байта', 'байт')}"
    st.caption(
        f"{used.count:,} целых чисел. ".replace(",", " ")
        + f"Здесь указатель занимает {pointer}, а объект int — {used.int_object_bytes}. "
        "В обычном 64-битном Python указатель занимает 8 байт, а int — 28, и тот же список занимает 3.4 МБ "
        "против 0.8 МБ у массива, т. е. в 4.5 раза больше. Массив от платформы не зависит: восемь байт на число."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Очистка данных в правильном порядке и векторизация. "
        "Порядок шагов очистки здесь — параметр, а не договорённость."
    )

    use_terms(
        st,
        "missing_values",
        "outlier",
        "mean_median",
        "vectorisation",
    )
    quick_start(
        st,
        purpose=(
            "Очистка данных: одни и те же шаги в разном порядке дают разный результат, причём неверный "
            "порядок не вызывает ошибки, а незаметно искажает таблицу."
        ),
        steps=(
            "Выберите в **«Когда считать статистику для заполнения»** вариант «до удаления выбросов», в **«Чем "
            "заполнять пропуски»** — «средним»: страница выделит невозможные значения красным.",
            "Замените «средним» на «медианой»: ошибка останется, но станет незаметной.",
            "Откройте вкладку «Векторизация и память»: она показывает, почему NumPy быстрее цикла.",
        ),
        notice="порядок шагов очистки — часть результата: неверный порядок не вызывает ошибки, а вносит искажение.",
    )
    how_to_read(
        st,
        what=(
            "Любой расчёт начинается с приведения данных в порядок: в них есть пропуски, "
            "дубликаты, разные написания одного и того же и явно невозможные значения. "
            "Здесь показано, что одни и те же шаги очистки, сделанные в разном порядке, дают "
            "разный результат, причём неверный порядок не вызывает ошибки, а незаметно вносит "
            "в таблицу новое искажение. Третья вкладка посвящена другому: почему те же вычисления "
            "на массивах NumPy идут в десятки раз быстрее, чем в цикле на Python."
        ),
        controls=(
            "Две главные настройки находятся вверху боковой панели. **«Когда считать статистику»** — до или после "
            "удаления невозможных значений; это и есть порядок шагов. **«Чем заполнять пропуски»** — "
            "средним или медианой. Остальные настройки меняют только данные второй вкладки (сколько строк, "
            "сколько в них пропусков и ошибок прибора) и размер замера на третьей."
        ),
        watch=(
            "Выберите «до удаления выбросов» и «средним»: очистка впишет в таблицу значения "
            "ниже физического предела, и страница сообщит об этом красным. Замените «средним» на «медианой» "
            "при том же порядке: ошибка останется, но станет незаметной, а это опаснее. "
            "На второй вкладке следите за «ошибкой очистки»: сколько дБм очистка прибавила "
            "к истинному среднему. На гистограмме оранжевый столбец — заполненные ячейки; "
            "если он стоит в стороне от остальных, искажение внесла сама очистка."
        ),
        data=(
            "Шесть строк первой вкладки — таблица с четырьмя намеренными дефектами. Вторая "
            "вкладка генерируется на месте: показания около −50 дБм, в которые подмешаны "
            "пропуски и −900 (код ошибки приёмника). Время и память измеряются "
            "по-настоящему, прямо в браузере."
        ),
    )
    reference(st, "missing_values", "outlier", "mean_median", "vectorisation")

    order, statistic, settings, size = _controls()
    small, larger, speed = st.tabs(["Шесть строк", "Тысяча строк", "Векторизация и память"])
    with small:
        _defect_tab(order, statistic)
    with larger:
        _readings_tab(settings, order, statistic)
    with speed:
        _speed_tab(size)

    with st.expander("Чего эта площадка не делает"):
        prose(st,
            "- Не заполняет пропуски по группам (медиана своего узла) и не интерполирует временной ряд: "
            "одна статистика на весь столбец.\n"
            "- Не ищет выбросы статистически: правило одно — физический предел −120 дБм. Значения, "
            "правдоподобные физически, но неверные, этим правилом не ловятся.\n"
            "- Не оптимизирует типы данных и не читает файл частями: это работа для ноутбука "
            "и настоящего файла.\n"
            "- Не сравнивает с вашим компьютером: замер делается в браузере, где Python медленнее."
        )


if __name__ == "__main__":
    main()
