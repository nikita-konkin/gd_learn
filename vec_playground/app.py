"""Streamlit-интерфейс playground'а векторизации текста."""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

from vec_playground.corpus import CONTENT_TYPES, texts_and_labels
from vec_playground.evaluation import (
    CLASSIFIERS,
    baseline_accuracy,
    confusion,
    evaluate,
    flipped,
    lab_comparison,
    mistakes,
    predictions,
    sweep_ngrams,
    top_features,
)
from vec_playground.features import ANALYZERS, FeatureSettings
from vec_playground.plotting import comparison_figure, confusion_figure, ngram_sweep_figure

# С чем сравнивается любая настройка: словарный TF-IDF без аргументов.
REFERENCE = FeatureSettings()
MAX_NGRAM = 6


@lru_cache(maxsize=1)
def _data():
    texts, labels = texts_and_labels()
    return texts, labels


@lru_cache(maxsize=1)
def _baseline() -> float:
    return baseline_accuracy(_data()[1])


@lru_cache(maxsize=64)
def _evaluate(settings: FeatureSettings, classifier: str):
    texts, labels = _data()
    return evaluate(texts, labels, settings, classifier)


@lru_cache(maxsize=64)
def _predictions(settings: FeatureSettings, classifier: str):
    texts, labels = _data()
    return tuple(predictions(texts, labels, settings, classifier))


@lru_cache(maxsize=8)
def _sweep(classifier: str):
    texts, labels = _data()
    return tuple(sweep_ngrams(texts, labels, classifier, max_n=MAX_NGRAM))


@lru_cache(maxsize=8)
def _lab_table(classifier: str):
    texts, labels = _data()
    return lab_comparison(texts, labels, classifier)


def _render_sidebar() -> tuple[FeatureSettings, str]:
    with st.sidebar:
        st.header("Признаки")
        analyzer = st.selectbox(
            "Из чего строим признаки",
            ANALYZERS,
            index=0,
            help="Слова, символьные n-граммы внутри слов или символы подряд через границы слов.",
        )
        character_mode = analyzer != "слова"
        low = 2 if character_mode else 1
        ngram_min, ngram_max = st.slider(
            "Длина n-граммы",
            1,
            MAX_NGRAM,
            (low, low) if character_mode else (1, 1),
            1,
            help="Для символов осмысленно начинать с 2-3.",
        )

        stemming = st.checkbox(
            "Стемминг",
            value=False,
            disabled=character_mode,
            help=(
                "Усекает слова до основы: «сохранить» и «сохранены» становятся одним "
                "признаком. Для символьных n-грамм неприменимо."
            ),
        )
        lowercase = st.checkbox("Приводить к нижнему регистру", value=True)
        min_df = st.slider(
            "min_df — выбросить признаки реже, чем в N текстах",
            1,
            5,
            1,
            1,
            help="Чистит словарь от случайных слов, но на маленьком корпусе режет и полезное.",
        )
        use_idf = st.checkbox(
            "Взвешивать по IDF",
            value=True,
            help="Выключите, чтобы получить обычный мешок слов со счётчиками.",
        )
        sublinear_tf = st.checkbox("Логарифмировать частоты (sublinear_tf)", value=False)

        st.header("Модель")
        classifier = st.selectbox("Классификатор", CLASSIFIERS, index=0)

        st.divider()
        st.caption(
            "Другие playground'ы: [градиентный спуск](../) · "
            "[метрики перевода](../mt/) · [языковая модель](../lm/)"
        )

    settings = FeatureSettings(
        analyzer=analyzer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        lowercase=lowercase,
        stemming=stemming,
        min_df=min_df,
        use_idf=use_idf,
        sublinear_tf=sublinear_tf,
    )
    return settings, classifier


def _render_score(settings: FeatureSettings, classifier: str) -> None:
    score = _evaluate(settings, classifier)
    baseline = _baseline()
    reference = _evaluate(REFERENCE, classifier)

    st.subheader("Точность")
    left, right = st.columns(2)
    left.metric(
        "Кросс-валидация, 5 частей",
        f"{score.accuracy:.3f}",
        delta=f"{score.accuracy - reference.accuracy:+.3f} к словарному TF-IDF",
    )
    right.metric("Признаков", f"{score.features}")
    st.caption(
        f"Разброс между частями ±{score.deviation:.3f} · baseline {baseline:.2f} · "
        f"{settings.describe()}"
    )

    if score.deviation > abs(score.accuracy - reference.accuracy) and settings != REFERENCE:
        st.info(
            "Разброс между частями больше, чем разница с эталонной настройкой. "
            "На 160 сегментах это обычное дело: такую разницу нельзя считать улучшением."
        )

    if score.accuracy <= baseline + 0.01:
        st.error("Модель не лучше выбора самого частого класса.")


def _render_mistakes(settings: FeatureSettings, classifier: str) -> None:
    texts, labels = _data()
    predicted = list(_predictions(settings, classifier))
    matrix = confusion(labels, predicted, list(CONTENT_TYPES))

    st.plotly_chart(confusion_figure(matrix), use_container_width=True)

    wrong = mistakes(texts, labels, predicted)
    st.caption(f"Ошибок: {len(wrong)} из {len(texts)}")
    st.dataframe(wrong, use_container_width=True, hide_index=True, height=240)


def _render_flips(settings: FeatureSettings, classifier: str) -> None:
    texts, labels = _data()
    if settings == REFERENCE:
        st.info(
            "Сейчас выбрана эталонная настройка. Поменяйте признаки в сайдбаре — "
            "здесь появятся сегменты, у которых из-за этого изменился ответ."
        )
        return

    changed = flipped(
        texts,
        labels,
        list(_predictions(REFERENCE, classifier)),
        list(_predictions(settings, classifier)),
    )
    if changed.empty:
        st.success("Ответы не изменились ни на одном сегменте.")
        return

    counts = changed["итог"].value_counts().to_dict()
    columns = st.columns(3)
    columns[0].metric("Исправлено", counts.get("исправлено", 0))
    columns[1].metric("Испорчено", counts.get("испорчено", 0))
    columns[2].metric("Всё ещё неверно", counts.get("всё ещё неверно", 0))
    st.caption(
        "Средняя точность прячет именно это: два набора признаков с одинаковым "
        "числом ошибаются на разных сегментах."
    )
    st.dataframe(changed, use_container_width=True, hide_index=True, height=280)


def _render_tabs(settings: FeatureSettings, classifier: str) -> None:
    tab_configs, tab_errors, tab_flips, tab_features = st.tabs(
        ["Конфигурации из работы", "Ошибки", "Что изменилось", "На что смотрит модель"]
    )

    with tab_configs:
        table = _lab_table(classifier)
        st.plotly_chart(
            comparison_figure(table, _baseline(), _evaluate(settings, classifier).accuracy),
            use_container_width=True,
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tab_errors:
        _render_mistakes(settings, classifier)

    with tab_flips:
        _render_flips(settings, classifier)

    with tab_features:
        texts, labels = _data()
        st.dataframe(top_features(texts, labels, settings), use_container_width=True, hide_index=True)
        st.caption(
            "Веса считаются логистической регрессией независимо от выбранного "
            "классификатора: вопрос «на что модель смотрит» один, а веса у "
            "наивного Байеса и SVM означают разное."
        )


def main():
    st.set_page_config(
        page_title="Векторизация текста",
        page_icon="abcd",
        layout="wide",
    )

    st.title("Текст → числа: признаки решают больше, чем модель")
    st.caption(
        "Классификация 160 сегментов локализации по типу контента. Крутите способ "
        "построения признаков и смотрите, что происходит с точностью — и какие "
        "сегменты из-за этого меняют ответ."
    )

    settings, classifier = _render_sidebar()

    left, right = st.columns([2, 3])
    with left:
        _render_score(settings, classifier)
    with right:
        st.plotly_chart(
            ngram_sweep_figure(list(_sweep(classifier)), _baseline()),
            use_container_width=True,
        )
        st.caption(
            "Словарные признаки упираются в потолок рано: слов в корпусе мало и "
            "они почти не повторяются. Символьные n-граммы ловят приставки, "
            "окончания и пунктуацию — для русского это решает."
        )

    _render_tabs(settings, classifier)


if __name__ == "__main__":
    main()
