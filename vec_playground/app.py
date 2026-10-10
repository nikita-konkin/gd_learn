"""Streamlit interface for the text-vectorisation playground."""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

from playground_common.explain import how_to_read, quick_start, reference
from playground_common.links import other_playgrounds
from playground_common.wording import as_printed
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

# What every setting is compared against: word TF-IDF with no arguments.
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
        st.caption(
            "Признаки — это числа, которыми описан текст. Модель видит только их, "
            "самого текста она не читает."
        )
        analyzer = st.selectbox(
            "На какие куски резать текст",
            ANALYZERS,
            index=0,
            help=(
                "Единица, из которой строятся признаки. «Слова» — текст делится по пробелам. "
                "Символьные варианты режут его на цепочки букв: внутри слова или подряд, "
                "через пробелы. Для русского буквы часто работают лучше слов — "
                "«локализация» и «локализации» это разные слова, но общий кусок «локализац»."
            ),
        )
        character_mode = analyzer != "слова"
        low = 2 if character_mode else 1
        ngram_min, ngram_max = st.slider(
            "Длина куска, n",
            1,
            MAX_NGRAM,
            (low, low) if character_mode else (1, 1),
            1,
            help=(
                "Сколько элементов подряд склеивать в один признак. Для слов 1 — отдельные слова, "
                "2 — пары соседних («не удалось»). Для букв осмысленно начинать с 2-3: "
                "одна буква встречается всюду и ни о чём не говорит."
            ),
        )

        stemming = st.checkbox(
            "Отсекать окончания (стемминг)",
            value=False,
            disabled=character_mode,
            help=(
                "Усекает слова до основы, чтобы формы одного слова стали одним признаком: "
                "«сохранить» и «сохранены» → «сохран». Для символьных кусков неприменимо — "
                "там понятия «слово» нет."
            ),
        )
        lowercase = st.checkbox(
            "Не различать заглавные и строчные",
            value=True,
            help="Если выключить, «Сохранить» и «сохранить» станут двумя разными признаками.",
        )
        min_df = st.slider(
            "Выбрасывать признаки, встретившиеся реже чем в N текстах",
            1,
            5,
            1,
            1,
            help=(
                "Параметр min_df. При 1 в словарь попадает всё, при 2 — выбрасывается то, "
                "что встретилось ровно один раз. Чистит словарь от случайного, но на корпусе "
                "в 160 сегментов режет и полезное: там почти всё встречается по разу."
            ),
        )
        use_idf = st.checkbox(
            "Понижать вес слов, которые есть почти везде (IDF)",
            value=True,
            help=(
                "С галочкой признаки взвешиваются по TF-IDF: редкое слово весит больше частого. "
                "Без неё остаётся «мешок слов» — просто счётчики вхождений."
            ),
        )
        sublinear_tf = st.checkbox(
            "Сглаживать частоты логарифмом",
            value=False,
            help=(
                "Параметр sublinear_tf. Десять вхождений слова считаются не в десять раз "
                "важнее одного, а примерно вдвое."
            ),
        )

        st.header("Модель")
        classifier = st.selectbox(
            "Классификатор",
            CLASSIFIERS,
            index=0,
            help=(
                "Алгоритм, который связывает признаки с ответом. Смысл страницы в том, "
                "что менять его почти бесполезно: признаки двигают точность в разы сильнее."
            ),
        )

        st.divider()
        st.caption(other_playgrounds("vec"))

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
        "Доля верных ответов",
        as_printed(score.accuracy),
        delta=f"{score.accuracy - reference.accuracy:+.3f} к настройке по умолчанию",
        help=(
            "Какая часть из 160 сегментов отнесена к правильному типу. Считается "
            "кросс-валидацией: корпус делится на 5 частей, модель по очереди учится "
            "на четырёх и проверяется на пятой, результат усредняется."
        ),
    )
    right.metric(
        "Признаков",
        f"{score.features}",
        help="Сколько разных слов или буквосочетаний попало в словарь при этих настройках.",
    )
    st.caption(
        f"Разброс между частями ±{score.deviation:.3f} — настолько результат гуляет "
        f"от части к части · точка отсчёта {baseline:.2f}: столько даёт ответ наугад "
        f"самым частым типом · {settings.describe()}"
    )

    if score.deviation > abs(score.accuracy - reference.accuracy) and settings != REFERENCE:
        st.info(
            "Разница с настройкой по умолчанию меньше, чем разброс между частями. "
            "На 160 сегментах это обычное дело: такую разницу нельзя считать улучшением — "
            "она целиком в пределах случайности разбиения."
        )

    if score.accuracy <= baseline + 0.01:
        st.error(
            "Модель не лучше точки отсчёта: она работает не точнее, чем ответ "
            "самым частым типом контента без чтения текста."
        )


def _render_mistakes(settings: FeatureSettings, classifier: str) -> None:
    texts, labels = _data()
    predicted = list(_predictions(settings, classifier))
    matrix = confusion(labels, predicted, list(CONTENT_TYPES))

    st.caption(
        "Матрица ошибок: строка — какой тип у сегмента на самом деле, столбец — что "
        "ответила модель. На диагонали верные ответы, вне её видно, кто с кем путается."
    )
    st.plotly_chart(confusion_figure(matrix), use_container_width=True)

    wrong = mistakes(texts, labels, predicted)
    st.caption(f"Ошибок: {len(wrong)} из {len(texts)}. Ниже — сами сегменты, на которых модель ошиблась.")
    st.dataframe(wrong, use_container_width=True, hide_index=True, height=240)


def _render_flips(settings: FeatureSettings, classifier: str) -> None:
    texts, labels = _data()
    if settings == REFERENCE:
        st.info(
            "Сейчас выбрана настройка по умолчанию — та, с которой всё сравнивается "
            "(слова, TF-IDF, без стемминга). Поменяйте что-нибудь в панели слева, и "
            "здесь появятся сегменты, у которых из-за этого изменился ответ модели."
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
    columns[0].metric(
        "Исправлено", counts.get("исправлено", 0),
        help="Было неверно при настройке по умолчанию, стало верно.",
    )
    columns[1].metric(
        "Испорчено", counts.get("испорчено", 0),
        help="Было верно, стало неверно.",
    )
    columns[2].metric(
        "Всё ещё неверно", counts.get("всё ещё неверно", 0),
        help="Ответ изменился, но правильным так и не стал.",
    )
    st.caption(
        "Средняя точность прячет именно это: два набора признаков могут ошибаться "
        "одинаковое число раз, но на разных сегментах."
    )
    st.dataframe(changed, use_container_width=True, hide_index=True, height=280)


def _render_tabs(settings: FeatureSettings, classifier: str) -> None:
    tab_configs, tab_errors, tab_flips, tab_features = st.tabs(
        [
            "Готовые настройки из лабораторной",
            "На чём модель ошиблась",
            "Какие сегменты поменяли ответ",
            "На что смотрит модель",
        ]
    )

    with tab_configs:
        st.caption(
            "Настройки, которые перебираются в лабораторной работе № 1, посчитанные здесь же. "
            "Пунктир — ответ наугад, сплошная черта — то, что выбрано у вас слева. "
            "Усы на столбиках — разброс между частями кросс-валидации."
        )
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

    quick_start(
        st,
        purpose=(
            "Машина работает с числами, а не с текстом. Страница показывает, как способ превратить текст в "
            "числа меняет точность на задаче «определить тип сегмента»."
        ),
        steps=(
            "Посмотрите на число **«Доля верных ответов»**: на словах 0.531 при точке отсчёта 0.25.",
            "Выберите в **«На какие куски резать текст»** «символы внутри слов», а **«Длина куска, n»** "
            "поставьте от 2 до 4: доля вырастет до 0.744.",
            "Смените **«Классификатор»** и сравните, что сдвигает точность сильнее — алгоритм или нарезка.",
        ),
        notice="разница меньше разброса между частями кросс-валидации — случайность, а не улучшение.",
    )
    how_to_read(
        st,
        what=(
            "Машина не умеет читать текст — она работает с числами. Эта страница про то, "
            "**как превратить текст в числа** и насколько сильно от этого зависит результат. "
            "Задача взята настоящая: по тексту сегмента определить, откуда он — из интерфейса "
            "программы, из документации, из рекламного текста или из юридического."
        ),
        controls=(
            "Всё в панели слева. Главный переключатель — верхний: резать текст на слова или "
            "на цепочки букв. Остальное — подробности этой нарезки. Внизу выбирается "
            "алгоритм; поменяйте его и сравните, что сдвигает точность сильнее — он или нарезка."
        ),
        watch=(
            "На «долю верных ответов» слева и на то, насколько она выше точки отсчёта 0.25. "
            "Разница меньше разброса между частями — не улучшение, а случайность. "
            "Во вкладках видно, **на каких именно** сегментах модель ошибается и какие из них "
            "меняют ответ, когда вы меняете признаки."
        ),
        data=(
            "160 сегментов из учебного корпуса курса, по 40 на каждый из четырёх типов "
            "контента. Тип проставлен человеком — это и есть разметка, по которой учится модель."
        ),
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

    reference(
        st,
        "accuracy",
        "baseline",
        "cross_validation",
        "fold_spread",
        "confusion_matrix",
        "bag_of_words",
        "tf_idf",
        "ngram",
        "stemming",
        "min_df",
        "sublinear_tf",
    )


if __name__ == "__main__":
    main()
