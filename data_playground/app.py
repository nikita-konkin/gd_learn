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
from playground_common.links import other_playgrounds
from playground_common.wording import plural

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
    st.sidebar.caption("Эти ручки меняют только порядок и способ очистки.")
    order = st.sidebar.radio(ORDER_LABEL, list(ORDERS), index=0)
    statistic = st.sidebar.radio(STATISTIC_LABEL, list(STATISTICS), index=0)

    st.sidebar.header("Тысяча строк")
    st.sidebar.caption("Эти ручки меняют только сами данные второй вкладки.")
    rows = st.sidebar.slider(ROWS_LABEL, 200, 5000, DEFAULT_READINGS.rows, step=200)
    missing = st.sidebar.slider(MISSING_LABEL, 0, 30, round(DEFAULT_READINGS.missing * 100))
    outliers = st.sidebar.slider(OUTLIERS_LABEL, 0, 15, round(DEFAULT_READINGS.outliers * 100))
    seed = st.sidebar.slider(SEED_LABEL, 0, 20, DEFAULT_READINGS.seed)

    st.sidebar.header("Скорость")
    st.sidebar.caption("Эта ручка меняет только размер замера.")
    size = st.sidebar.select_slider(SIZE_LABEL, SIZES, value=SUM_SIZE)

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/data"))

    settings = ReadingsSettings(rows=rows, missing=missing / 100, outliers=outliers / 100, seed=seed)
    return order, statistic, settings, size


def _cleaned_view(cleaned: Cleaned) -> pd.DataFrame:
    view = cleaned.table.copy()
    view["сигнал"] = view["сигнал"].round(2)
    view["заполнено"] = ["← заполнено" if flag else "" for flag in cleaned.filled]
    return view


def _defect_tab(order: str, statistic: str) -> None:
    raw = defect_table()
    found = inspect(raw)
    st.markdown(
        "Шесть показаний узлов связи с четырьмя намеренными дефектами: пропуски, дубликат, "
        "разнобой в регистре и физически невозможное значение."
    )
    st.dataframe(raw, use_container_width=True, hide_index=True)

    left, middle, right = st.columns(3)
    left.metric("Пропусков", str(sum(found.missing.values())))
    middle.metric("Полных дубликатов", str(found.duplicates))
    right.metric("Вариантов статуса", str(len(found.statuses)))
    st.caption(
        f"Пропуски по столбцам: {', '.join(f'{column} — {count}' for column, count in found.missing.items())}. "
        f"Статусы: {', '.join(found.statuses)} — «ok» и «OK» означают одно и то же."
    )

    cleaned = clean_defect_table(statistic, order)
    st.subheader("После очистки")
    st.dataframe(_cleaned_view(cleaned), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    left.metric("Заполнено значением", f"{cleaned.fill_value:.2f}")
    right.metric("Невозможных значений после очистки", str(cleaned.impossible_left))

    if (order, statistic) == (AFTER, MEDIAN):
        st.info(
            "Правильный порядок: выбросы сначала превращены в пропуски, потом все пропуски заполнены "
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
            "вместо −48.65. Ошибка есть, но она не бросается в глаза — и потому опаснее."
        )


def _readings_tab(settings: ReadingsSettings, order: str, statistic: str) -> None:
    readings = _readings(settings)
    result = _estimate(settings, statistic, order)
    st.markdown(
        f"{settings.rows} показаний трёх узлов, истинный уровень около −50 дБм. В них заложено "
        f"{int(readings.missing_mask.sum())} пропусков и {int(readings.outlier_mask.sum())} значений −900."
    )

    left, middle, right = st.columns(3)
    left.metric("Истинное среднее", f"{result.true_mean:.2f}")
    middle.metric("Среднее после очистки", f"{result.cleaned_mean:.2f}")
    right.metric("Ошибка очистки", f"{result.error:+.2f}")

    cleaned = clean_signal(readings.table, statistic, order)
    st.plotly_chart(
        cleaned_histogram(cleaned.table["сигнал"].to_numpy(), np.asarray(cleaned.filled, dtype=bool), result.true_mean),
        use_container_width=True,
    )
    st.caption(
        f"Все {result.filled} пропусков заполнены одним значением {result.fill_value:.2f}. "
        "Оранжевый столбец — это они. Если он стоит внутри распределения, заполнение ничего не исказило; "
        "если отдельно от него — исказила сама очистка."
    )
    if result.impossible_left:
        st.error(
            f"Значение заполнения {result.fill_value:.2f} ниже физического предела {OUTLIER_LIMIT:.0f}: "
            f"очистка создала {result.impossible_left} невозможных показаний."
        )


def _speed_tab(size: int) -> None:
    if st.button(REMEASURE_LABEL):
        _timings.cache_clear()
    timings = _timings(size)
    slowest = timings[0].seconds
    spaced = f"{size:,}".replace(",", " ")

    st.markdown(
        f"Сумма квадратов {spaced} чисел тремя способами. Различаются они не алгоритмом, а тем, кто выполняет цикл."
    )
    st.plotly_chart(timing_figure(timings), use_container_width=True)
    columns = st.columns(len(timings))
    for column, timing in zip(columns, timings, strict=True):
        speedup = slowest / timing.seconds if timing.seconds else float("inf")
        delta = f"×{speedup:.0f} быстрее" if timing is not timings[0] else None
        column.metric(timing.method, f"{timing.seconds * 1000:.1f} мс", delta)
    st.metric("Результаты совпадают", "да" if results_agree(timings) else "нет")
    st.caption(
        "Время измерено только что, в этом браузере. На вашем компьютере в обычном Python оно будет другим: "
        "здесь Python работает внутри WebAssembly. Порядок величин "
        f"остаётся тем же, как и причина, по которой «{DOT}» обгоняет поэлементный вариант: он не создаёт "
        "промежуточный массив квадратов."
    )

    st.subheader("Память: список против массива")
    used = _memory()
    left, middle, right = st.columns(3)
    left.metric("Список, МБ", f"{used.list_bytes / 1024**2:.1f}")
    middle.metric("Массив int64, МБ", f"{used.array_bytes / 1024**2:.1f}")
    right.metric("Разница, раз", f"{used.ratio:.1f}")
    pointer = f"{used.pointer_bytes} {plural(used.pointer_bytes, 'байт', 'байта', 'байт')}"
    st.caption(
        f"{used.count:,} целых чисел. ".replace(",", " ")
        + f"Здесь указатель занимает {pointer}, а объект int — {used.int_object_bytes}. "
        "В обычном 64-битном Python указатель — 8 байт, а int — 28, и тот же список занимает 3.4 МБ "
        "против 0.8 МБ у массива, в 4.5 раза больше. Массив от платформы не зависит: восемь байт на число всегда."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Очистка данных в правильном порядке и векторизация. "
        "Порядок шагов очистки здесь — параметр, а не договорённость."
    )

    order, statistic, settings, size = _controls()
    small, larger, speed = st.tabs(["Шесть строк", "Тысяча строк", "Векторизация и память"])
    with small:
        _defect_tab(order, statistic)
    with larger:
        _readings_tab(settings, order, statistic)
    with speed:
        _speed_tab(size)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
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
