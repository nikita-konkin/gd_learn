"""Streamlit interface for the translation-memory playground."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import streamlit as st

from tm_playground.clustering import agreement_table, cluster, corpus_map, crosstab
from tm_playground.corpus import (
    EMBEDDING_MODEL,
    load_corpus,
    load_queries,
    load_query_vectors,
    load_segment_vectors,
)
from tm_playground.measures import (
    CHAR_NGRAMS,
    DEFAULT_THRESHOLD,
    LEVENSHTEIN,
    LSA,
    MEASURE_HELP,
    MEASURE_LABELS,
    MEASURE_UNITS,
    MEASURES,
)
from tm_playground.plotting import agreement_figure, divergence_figure, map_figure, threshold_figure
from tm_playground.search import (
    SearchIndex,
    comparison_table,
    divergences,
    suggestions,
    threshold_sweep,
)

LAB_QUERIES = "Запрос из работы"
OWN_QUERY = "Свой текст"


@lru_cache(maxsize=1)
def _index() -> SearchIndex:
    return SearchIndex(load_corpus(), load_queries(), load_segment_vectors(), load_query_vectors())


@lru_cache(maxsize=256)
def _scores(query: str, measure: str):
    """Cached because Levenshtein against 160 segments is real work in a browser."""
    return _index().scores(query, measure)


@lru_cache(maxsize=1)
def _comparison() -> pd.DataFrame:
    return comparison_table(_index())


@lru_cache(maxsize=1)
def _sweep() -> pd.DataFrame:
    return threshold_sweep(_index())


@lru_cache(maxsize=1)
def _agreement() -> pd.DataFrame:
    index = _index()
    return agreement_table(index.representations(), index.corpus["type"])


@lru_cache(maxsize=1)
def _map():
    index = _index()
    return corpus_map(index.representations()[LSA])


def _render_sidebar() -> tuple[str, int, float]:
    index = _index()
    with st.sidebar:
        st.header("Запрос")
        source = st.radio(
            "Что ищем в памяти",
            (LAB_QUERIES, OWN_QUERY),
            index=0,
            help="Запросы из работы уже имеют посчитанные эмбеддинги, свой текст — нет.",
        )
        if source == LAB_QUERIES:
            options = list(index.queries["text"])
            notes = dict(zip(index.queries["text"], index.queries["note"], strict=True))
            query = st.selectbox(
                "Сегмент на перевод",
                options,
                index=0,
                format_func=lambda text: f"{text}  ({notes[text]})" if notes[text] else text,
            )
        else:
            # A single-line input on purpose: st.text_area only applies its
            # contents on Ctrl+Enter, so clicking away leaves the page showing a
            # confident answer to the previous query. text_input commits on blur.
            query = st.text_input(
                "Сегмент на перевод",
                value="Сохранить настройки поиска",
                help="Любой русский текст. Эмбеддинги для него посчитать негде — см. результаты.",
            )

        st.header("Поиск")
        top_n = st.slider("Сколько подсказок показывать", 1, 5, 3, 1)
        threshold = st.slider(
            "Порог совпадения, %",
            50,
            95,
            int(DEFAULT_THRESHOLD),
            5,
            help="Ниже порога подсказка не показывается. Это решение человека о цене ошибки, а не свойство меры.",
        )

        st.divider()
        st.caption(
            "Другие playground'ы: [градиентный спуск](../) · "
            "[метрики перевода](../mt/) · [языковая модель](../lm/) · "
            "[векторизация](../vec/) · [данные и разметка](../labels/)"
        )
    return str(query).strip(), int(top_n), float(threshold)


def _results_table(query: str, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    """One row per measure and rank, plus the measures that could not answer."""
    index = _index()
    rows: list[dict[str, object]] = []
    unavailable: list[str] = []

    for measure in MEASURES:
        if not index.supports(query, measure):
            unavailable.append(MEASURE_LABELS[measure])
            continue
        for hit in index.search(query, measure, top_n):
            digits = 1 if measure == LEVENSHTEIN else 3
            rows.append(
                {
                    "мера": MEASURE_LABELS[measure],
                    "ранг": hit.rank,
                    "сегмент из памяти": hit.segment,
                    "оценка": f"{hit.score:.{digits}f}{MEASURE_UNITS[measure]}",
                    "тип контента": hit.content_type,
                }
            )
    return pd.DataFrame(rows), unavailable


def _render_query(query: str, top_n: int, threshold: float) -> None:
    index = _index()
    if not query:
        st.info("Введите сегмент в сайдбаре.")
        return

    best_levenshtein = index.best(query, LEVENSHTEIN)
    best_cosine = index.best(query, CHAR_NGRAMS)
    agree = best_levenshtein.segment == best_cosine.segment

    if agree:
        st.success(
            f"**Меры согласны.** Обе выбрали «{best_levenshtein.segment}» — "
            f"{best_levenshtein.score:.0f}% по Левенштейну, {best_cosine.score:.3f} по косинусу."
        )
    else:
        st.warning(
            f"**Меры разошлись.** Левенштейн предлагает «{best_levenshtein.segment}» "
            f"({best_levenshtein.score:.0f}%), косинус — «{best_cosine.segment}» "
            f"({best_cosine.score:.3f}). Какая подсказка полезнее переводчику, "
            "решает человек: это и есть главное задание работы."
        )

    left, right = st.columns([3, 4])
    with left:
        table, unavailable = _results_table(query, top_n)
        st.dataframe(table, use_container_width=True, hide_index=True, height=min(60 + 35 * len(table), 460))
        if unavailable:
            st.info(
                f"**{', '.join(unavailable)}** для своего текста посчитать негде. "
                f"Чтобы превратить строку в такой вектор, нужна сама модель "
                f"({EMBEDDING_MODEL}) и около 500 МБ зависимостей — в браузере их нет. "
                "Готовые векторы посчитаны заранее только для запросов из работы. "
                "Это не пробел playground'а: именно поэтому эмбеддинги в проекте — "
                "отдельный этап конвейера, а не строка в ноутбуке."
            )
        if best_levenshtein.score < threshold:
            st.caption(
                f"При пороге {threshold:.0f}% переводчик не увидел бы ничего: "
                f"лучшее совпадение — {best_levenshtein.score:.0f}%."
            )

    with right:
        st.plotly_chart(
            divergence_figure(
                _scores(query, LEVENSHTEIN),
                _scores(query, CHAR_NGRAMS),
                index.segments,
                threshold,
            ),
            use_container_width=True,
        )
        st.caption(
            "Каждая точка — сегмент памяти. Если бы меры мерили одно и то же, "
            "облако легло бы на диагональ. Точки слева вверху косинус считает "
            "близкими, а Левенштейн — нет: это перестановки слов и короткие "
            "сегменты внутри длинных запросов."
        )


def _render_all_queries() -> None:
    table = _comparison()
    diverging = divergences(table)
    agreed = int(table["agree"].sum())

    columns = st.columns(2)
    columns[0].metric("Меры выбрали одно и то же", f"{agreed} из {len(table)}")
    columns[1].metric("Разошлись", f"{len(diverging)}")

    display = pd.DataFrame(
        {
            "запрос": table["query"],
            "что проверяет": table["note"],
            "Левенштейн": table["levenshtein_hit"],
            "%": table["levenshtein_score"],
            "косинус": table["char_ngrams_hit"],
            "близость": table["char_ngrams_score"],
            "эмбеддинги": table["embeddings_hit"],
            "совпали": table["agree"],
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True, height=340)
    st.caption(
        "Левенштейн считает символы и чувствителен к длине и порядку. Косинус "
        "нормирует на длину и порядок игнорирует, но легко обманывается "
        "канцелярской лексикой. Ни одна из мер не «правильная»: правильная — "
        "та, чьи ошибки дешевле для конкретного проекта."
    )


def _render_threshold(threshold: float) -> None:
    st.plotly_chart(threshold_figure(_sweep(), threshold), use_container_width=True)

    table = suggestions(_index(), threshold)
    offered = int(table["offered"].sum())
    st.caption(
        f"При пороге {threshold:.0f}% подсказку получают {offered} запросов из {len(table)}. "
        "Порог — это выбор между лишней работой по правке негодной подсказки и "
        "потерянной подсказкой, которая пригодилась бы."
    )
    display = pd.DataFrame(
        {
            "запрос": table["query"],
            "что проверяет": table["note"],
            "лучший сегмент памяти": table["suggestion"],
            "%": table["percent"],
            "показана": table["offered"],
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True, height=340)


def _render_clustering() -> None:
    index = _index()
    table = _agreement()
    st.plotly_chart(agreement_figure(table), use_container_width=True)
    st.caption(
        "Усы — разброс по восьми начальным приближениям. У символьных n-грамм "
        "он шире самой разницы между мерами, так что одно число ARI с тремя "
        "знаками после запятой читать как результат нельзя."
    )
    st.warning(
        "**Отрицательный результат, и он главный.** Смена представления сдвигает "
        "ARI, но задачу не решает, потому что задача поставлена неверно: "
        "кластеризация ищет группы, которые есть в данных, а нам нужны группы, "
        "которые придумал человек. Совпадать они не обязаны ни при каком "
        "представлении. Обучение без учителя не заменяет разметку."
    )
    st.dataframe(
        crosstab(index.corpus["type"], cluster(index.representations()[CHAR_NGRAMS])),
        use_container_width=True,
    )


def _render_map() -> None:
    index = _index()
    st.plotly_chart(map_figure(_map(), index.corpus["type"]), use_container_width=True)
    st.caption(
        "Две компоненты сохраняют лишь часть изменчивости: перекрытие точек на "
        "плоскости не означает, что сегменты неразличимы в исходном "
        "пространстве. Карта — инструмент для гипотез, а не доказательство."
    )


def main():
    st.set_page_config(page_title="Память переводов", page_icon="🔎", layout="wide")

    st.title("Память переводов: чем мерить похожесть")
    st.caption(
        "Нечёткий поиск по памяти переводов — та самая подсказка «совпадение 87%» "
        "из Trados и memoQ. Это поиск ближайшего соседа, и весь вопрос в том, что "
        "считать расстоянием между двумя текстами. Разные ответы дают разные "
        "подсказки — и ошибаются по-разному."
    )

    query, top_n, threshold = _render_sidebar()
    _render_query(query, top_n, threshold)

    tab_queries, tab_threshold, tab_clusters, tab_map = st.tabs(
        ["Все запросы работы", "Порог", "Кластеризация", "Карта корпуса"]
    )
    with tab_queries:
        _render_all_queries()
    with tab_threshold:
        _render_threshold(threshold)
    with tab_clusters:
        _render_clustering()
    with tab_map:
        _render_map()

    with st.expander("Чем меры отличаются"):
        st.dataframe(
            pd.DataFrame(
                {
                    "мера": [MEASURE_LABELS[measure] for measure in MEASURES],
                    "как устроена": [MEASURE_HELP[measure] for measure in MEASURES],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
