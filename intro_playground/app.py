"""Streamlit interface for the intro-lecture playground."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import streamlit as st

from intro_playground.corpus import CONTENT_TYPES, load_corpus
from intro_playground.length import (
    RULE_OF_THUMB,
    expansion_by_type,
    fit_line,
    lengths,
    mean_expansion,
    overflow_share,
    total_expansion,
)
from intro_playground.pairs import LECTURE_MIN_COUNT, LECTURE_TOP, pair_table, top_by_count, top_by_pmi
from intro_playground.plotting import ASH, PMI_COLOR, depth_figure, length_figure, overflow_figure, pairs_figure
from intro_playground.tree import (
    DEPTHS,
    FEATURE_LABELS,
    FEATURES,
    FINAL_PERIOD,
    LECTURE_DEPTH,
    NGRAM_ACCURACY,
    build_tree,
    depth_sweep,
    feature_table,
    tree_accuracy,
    tree_nodes,
    used_features,
)
from playground_common.links import other_playgrounds
from playground_common.wording import plural, segments

BASELINE = 0.25  # four balanced content types: always answering one of them is right a quarter of the time


@lru_cache(maxsize=1)
def _corpus() -> pd.DataFrame:
    return load_corpus()


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    return feature_table(_corpus())


@lru_cache(maxsize=64)
def _sweep(chosen: tuple[str, ...]) -> pd.DataFrame:
    """Six trees of five fits each; cached because the browser redoes them on every rerun."""
    return depth_sweep(_features()[list(chosen)], _corpus()["type"])


@lru_cache(maxsize=1)
def _lengths() -> pd.DataFrame:
    return lengths(_corpus())


@lru_cache(maxsize=1)
def _pairs() -> pd.DataFrame:
    return pair_table(_corpus()["ru_ref"])


def _percent(value: float) -> str:
    return f"{value:+.0%}"


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Что здесь")
        st.markdown(
            "Три задачи из лекции «Введение в машинное обучение», для которых нет "
            "отдельной лабораторной:\n\n"
            "1. **классификация** деревом решений на признаках, придуманных лингвистом;\n"
            "2. **регрессия**: сколько места займёт перевод;\n"
            "3. **поиск правил**: какие слова ходят парами — заготовка глоссария."
        )
        st.caption(
            "Корпус — те же 160 сегментов локализации, что в лабораторных. Числа по "
            "умолчанию совпадают со слайдами лекции."
        )
        st.divider()
        st.caption(other_playgrounds("intro"))


def _rules_markdown(nodes) -> str:
    lines = []
    for node in nodes:
        indent = "  " * node.level
        if node.question is not None:
            text = f"**{node.question}**"
        else:
            text = f"{node.verdict} — {segments(node.segments)}, {node.purity:.0%} этого типа"
        lines.append(f"{indent}- {node.branch + ' → ' if node.branch else ''}{text}")
    return "\n".join(lines)


def _render_tree() -> None:
    chosen = st.multiselect(
        "Признаки",
        FEATURES,
        default=list(FEATURES),
        format_func=FEATURE_LABELS.get,
        help="Шесть признаков со слайда лекции. Уберите любой и посмотрите, чем дерево его заменит.",
    )
    depth = st.slider(
        "Глубина дерева", DEPTHS[0], DEPTHS[-1], LECTURE_DEPTH, 1, help="Сколько вопросов подряд можно задать."
    )
    if not chosen:
        st.warning("Выберите хотя бы один признак: без признаков дереву не о чем спрашивать.")
        return

    # Keep the lecture's column order whatever order the student picked them in.
    chosen = tuple(name for name in FEATURES if name in chosen)
    corpus, features = _corpus(), _features()[list(chosen)]
    accuracy = float(_sweep(chosen).set_index("depth").loc[depth, "accuracy"])
    tree = build_tree(depth).fit(features, corpus["type"])

    columns = st.columns(3)
    columns[0].metric("Точность дерева", f"{accuracy:.3f}")
    columns[1].metric("n-граммы, Л.р. № 1", f"{NGRAM_ACCURACY:.3f}")
    columns[2].metric("Самый частый класс", f"{BASELINE:.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Правила, которые вывело дерево**")
        st.markdown(_rules_markdown(tree_nodes(tree, list(chosen))))
    with right:
        st.plotly_chart(depth_figure(_sweep(chosen), depth, BASELINE, NGRAM_ACCURACY), use_container_width=True)

    asked = used_features(tree, list(chosen))
    idle = [name for name in chosen if name not in asked]
    # Labels carry their own guillemets («вы», «ваш»), so they are listed bare, separated by semicolons.
    message = "Дерево спросило про: " + "; ".join(FEATURE_LABELS[name] for name in asked) + "."
    if idle:
        message += (
            " Не понадобились: " + "; ".join(FEATURE_LABELS[name] for name in idle) + ". "
            "Признак может быть верным и всё равно лишним — если его работу уже сделал другой."
        )
    st.caption(message)

    if FINAL_PERIOD not in chosen:
        full = tree_accuracy(_features(), corpus["type"], depth)
        st.info(
            f"**Без финальной точки точность {accuracy:.3f} вместо {full:.3f}.** Интерфейсные строки "
            "не кончаются точкой — один этот признак отделяет весь интерфейс. Посмотрите, что дерево "
            "взяло взамен."
        )
    st.caption(
        f"Точность дерева ниже, чем у символьных n-грамм ({NGRAM_ACCURACY:.3f}), но его решения можно "
        "прочесть и показать заказчику. Выбор между понятностью и точностью — условие договора, а не "
        "техническая мелочь."
    )

    with st.expander("Как посчитаны признаки и где они срабатывают"):
        table = (
            _features()
            .assign(type=corpus["type"])
            .groupby("type")
            .agg({name: "mean" if name == "words" else "sum" for name in FEATURES})
        )
        table = table.loc[list(CONTENT_TYPES)].T
        table.index = [FEATURE_LABELS[name] for name in table.index]
        st.dataframe(table.round(1), use_container_width=True)
        st.caption(
            "Для длины — среднее число слов, для остальных — сколько сегментов типа имеют признак "
            "(из 40). Плейсхолдер ищется в английском оригинале, остальное — в эталонном переводе."
        )


def _render_length() -> None:
    table, ids = _lengths(), _corpus()["id"]
    line = fit_line(table)
    by_type = expansion_by_type(table, CONTENT_TYPES).set_index("type")

    columns = st.columns(3)
    columns[0].metric("R² прямой", f"{line.r2:.2f}")
    columns[1].metric("Длиннее на сегмент", _percent(mean_expansion(table)))
    columns[2].metric("Длиннее в сумме", _percent(total_expansion(table)))

    st.plotly_chart(length_figure(table, line, ids), use_container_width=True)
    widest, narrowest = by_type["mean"].idxmax(), by_type["mean"].idxmin()
    st.caption(
        f"Отраслевое правило — «русский длиннее на {RULE_OF_THUMB:.0%}». На нашем корпусе в среднем "
        f"{_percent(mean_expansion(table))} на сегмент и {_percent(total_expansion(table))} в сумме символов: "
        "длинные сегменты растут меньше коротких. А разброс по типам больше самой поправки: "
        f"{widest} {_percent(by_type.loc[widest, 'mean'])}, {narrowest} {_percent(by_type.loc[narrowest, 'mean'])}."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "тип": by_type.index,
                "в среднем": [_percent(value) for value in by_type["mean"]],
                "медиана": [_percent(value) for value in by_type["median"]],
                "в сумме символов": [_percent(value) for value in by_type["total"]],
                "влезает 9 из 10 при запасе": [_percent(value) for value in by_type["high"]],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Сколько места оставить под перевод")
    margin = st.slider(
        "Запас под перевод, %",
        0,
        60,
        int(RULE_OF_THUMB * 100),
        5,
        help="Строка не влезает, если перевод длиннее оригинала больше чем на столько процентов.",
    )
    shares = overflow_share(table, margin / 100, CONTENT_TYPES)
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(overflow_figure(shares, margin / 100), use_container_width=True)
    with right:
        interface = shares["интерфейс"]
        count = round(interface * by_type.loc["интерфейс", "segments"])
        st.metric("Строк интерфейса не влезает", f"{interface:.0%}")
        st.caption(
            f"При запасе {margin} % не влезает {count} {plural(count, 'строка', 'строки', 'строк')} "
            f"интерфейса из {by_type.loc['интерфейс', 'segments']:.0f}. Средний коэффициент, применённый "
            "ко всему проекту, ошибается там, где цена ошибки выше всего, — на кнопках."
        )

        source = st.text_input("Английская строка", "Save changes")
        content_type = st.selectbox("Тип контента", CONTENT_TYPES)
        size = len(source)
        row = by_type.loc[content_type]
        st.markdown(
            f"Оригинал — **{size}** симв. По общей прямой перевод займёт **{line.predict(size):.0f}**, "
            f"по среднему для типа «{content_type}» — **{size * (1 + row['mean']):.0f}**. "
            f"Чтобы влезло в 9 случаях из 10, оставьте **{size * (1 + row['high']):.0f}**."
        )
        known = _corpus()[_corpus()["en"] == source.strip()]
        if not known.empty:
            translation = str(known.iloc[0]["ru_ref"])
            st.caption(
                f"Эта строка есть в корпусе: «{translation}» — {len(translation)} симв. "
                "Прогноз — это ставка, а не обещание."
            )


def _render_pairs() -> None:
    table = _pairs()
    min_count = st.slider(
        "Минимум вхождений пары",
        1,
        5,
        LECTURE_MIN_COUNT,
        1,
        help="Пары, встреченные реже, в рейтинг не попадают. На слайде — 2.",
    )
    top = st.slider("Сколько пар показать", 5, 15, LECTURE_TOP, 1)

    by_count, by_pmi = top_by_count(table, min_count, top), top_by_pmi(table, min_count, top)
    kept = int((table["count"] >= min_count).sum())
    shared = len(set(by_count["pair"]) & set(by_pmi["pair"]))

    columns = st.columns(3)
    columns[0].metric("Разных пар слов", len(table))
    columns[1].metric(f"Не реже {min_count} {plural(min_count, 'раза', 'раз', 'раз')}", kept)
    columns[2].metric("Общих в двух списках", f"{shared} из {min(top, len(by_pmi))}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            pairs_figure(by_count, "count", "Просто по частоте", ASH, "сколько раз встретилась"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            pairs_figure(by_pmi, "pmi", "По силе связи (PMI)", PMI_COLOR, "PMI, бит"), use_container_width=True
        )

    if min_count == 1:
        st.warning(
            "**PMI любит редкое.** Пара, встреченная один раз из двух редких слов, получает наибольший "
            "PMI, какой вообще возможен, — и весь верх рейтинга занимают случайные соседства. Поэтому "
            "порог по числу вхождений обязателен."
        )
    else:
        st.caption(
            f"На 160 коротких сегментах хотя бы {min_count} "
            f"{plural(min_count, 'раз', 'раза', 'раз')} повторяются всего {kept} "
            f"{plural(kept, 'пара', 'пары', 'пар')} из {len(table)}, поэтому оба рейтинга перебирают почти "
            "одно и то же. По частоте наверх идут служебные обороты («в течение», «по умолчанию»); PMI "
            "поднимает то, что ходит парами: «программное обеспечение», «настоящие условия». Это заготовка "
            "глоссария — её остаётся вычитать лингвисту."
        )

    with st.expander("Все пары выше порога"):
        rows = table[table["count"] >= min_count].sort_values("pmi", ascending=False, kind="stable")
        st.dataframe(
            rows.rename(columns={"pair": "пара", "count": "вхождений", "pmi": "PMI, бит"}).round(2),
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(page_title="Три задачи вводной лекции", page_icon="🌳", layout="wide")

    st.title("Дерево, регрессия, правила: три задачи вводной лекции")
    st.caption(
        "Классификация, регрессия и поиск правил на корпусе курса — задачи, которые лекция разбирает "
        "сама, без лабораторной. Здесь их можно покрутить."
    )

    _render_sidebar()

    tab_tree, tab_length, tab_pairs = st.tabs(["Дерево решений", "Длина перевода", "Сочетания для глоссария"])
    with tab_tree:
        _render_tree()
    with tab_length:
        _render_length()
    with tab_pairs:
        _render_pairs()
