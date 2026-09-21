"""Streamlit interface for the data-and-annotation playground."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from labels_playground.agreement import (
    CATEGORIES,
    agreement,
    agreement_matrix,
    chance_agreement,
    chance_sweep,
    expected_raw_agreement,
    kappa_reading,
)
from labels_playground.corpus import load_annotations, load_corpus, load_newsgroups_curve, load_remedies
from labels_playground.learning import (
    CHARACTERS,
    CLASSIFIER_LABELS,
    CURVE_CLASSIFIERS,
    FEATURE_LABELS,
    FEATURE_SETS,
    LOGISTIC,
    WORDS,
    Curve,
    baseline_accuracy,
    catch_up_size,
    empty_vectors,
    learning_curve_for,
)
from labels_playground.mqm import (
    CRITICAL,
    MISSED_CRITICAL,
    NONE,
    REJECTED_CORRECT,
    SEVERITIES,
    SEVERITY_WEIGHTS,
    SEVERITY_WHEN,
    acceptance,
    acceptance_errors,
    acceptance_sweep,
    clean_thresholds,
    lab_flags,
    penalties,
    rank_correlation,
)
from labels_playground.plotting import (
    acceptance_figure,
    chance_figure,
    learning_figure,
    matrix_figure,
    mqm_figure,
    newsgroups_figure,
)
from labels_playground.wording import segments

REMEDY_LABELS = {
    "hyperparameters": ("подобрать гиперпараметры", "значения по умолчанию уже оптимальны"),
    "feature_union": ("склеить словарные и символьные признаки", "больше признаков при малой выборке — это шум"),
    "more_data": ("разметить больше данных", "единственное, что работает"),
}

THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


@lru_cache(maxsize=1)
def _corpus() -> pd.DataFrame:
    return load_corpus()


@lru_cache(maxsize=1)
def _annotations() -> pd.DataFrame:
    return load_annotations()


@lru_cache(maxsize=8)
def _curve(feature_set: str, classifier: str) -> Curve:
    """Thirty fits each; cached because the browser redoes them on every rerun."""
    corpus = _corpus()
    return learning_curve_for(corpus["ru_ref"], corpus["type"], feature_set, classifier)


@lru_cache(maxsize=2)
def _empty_vectors(feature_set: str) -> int:
    corpus = _corpus()
    return empty_vectors(corpus["ru_ref"], corpus["type"], feature_set)


@lru_cache(maxsize=1)
def _baseline() -> float:
    return baseline_accuracy(_corpus()["type"])


@lru_cache(maxsize=1)
def _median_bleu() -> float:
    return float(_corpus()["bleu"].median())


def _sentence(label: str) -> str:
    """Upper-case the first letter only: ``str.capitalize`` would turn TF-IDF into tf-idf."""
    return label[:1].upper() + label[1:]


def _short(text: str, width: int = 34) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Что здесь")
        st.markdown(
            "Три вопроса блока 5 лекции «Текст как данные», каждый на своей вкладке:\n\n"
            "1. во что упёрлось качество — в модель или в данные;\n"
            "2. можно ли принимать перевод по BLEU;\n"
            "3. насколько можно верить самой разметке."
        )
        st.caption(
            "Числа — из Л.р. № 1 (разделы 9, 12, 13) и Л.р. № 3 (разделы 6, 7). "
            "Кривые обучения на нашем корпусе считаются прямо здесь; кривая 20 Newsgroups — "
            "в ноутбуке, потому что корпус весит 14 МБ."
        )
        st.divider()
        st.caption(
            "Другие playground'ы: [градиентный спуск](../) · [метрики перевода](../mt/) · "
            "[языковая модель](../lm/) · [векторизация](../vec/) · [память переводов](../tm/)"
        )


def _render_data_or_model() -> None:
    classifier = st.selectbox(
        "Классификатор",
        CURVE_CLASSIFIERS,
        index=CURVE_CLASSIFIERS.index(LOGISTIC),
        format_func=CLASSIFIER_LABELS.get,
        help=(
            "Модели раздела 9 Л.р. № 1. Третьей, ближайших соседей, здесь нет: на словах у пустого "
            "вектора все соседи на одном расстоянии, и какие пять из них «ближайшие», решает округление."
        ),
    )
    curves = {feature_set: _curve(feature_set, classifier) for feature_set in FEATURE_SETS}
    words, characters = curves[WORDS], curves[CHARACTERS]
    catch_up = catch_up_size(characters, words)
    full = int(words.sizes[-1])

    columns = st.columns(3)
    columns[0].metric(f"{_sentence(FEATURE_LABELS[WORDS])}, {segments(full)}", f"{words.final:.3f}")
    columns[1].metric(f"{_sentence(FEATURE_LABELS[CHARACTERS])}, {segments(full)}", f"{characters.final:.3f}")
    columns[2].metric(
        "Символьные догоняют слова на объёме",
        segments(catch_up) if catch_up is not None else "не догоняют",
    )

    st.plotly_chart(learning_figure(curves, _baseline(), catch_up), use_container_width=True)

    if catch_up is not None:
        st.caption(
            "Кривые читаются и по горизонтали. Символьные n-граммы догоняют слова, "
            f"обученные на всех {full}, уже при объёме {catch_up}: удачная идея о признаках "
            f"экономит {segments(full - catch_up)} разметки. Это единственное место, где "
            "признаки меняются на данные напрямую."
        )
    empty_words, empty_characters = _empty_vectors(WORDS), _empty_vectors(CHARACTERS)
    st.caption(
        f"Почему слова проигрывают. У {empty_words} из {len(_corpus())} предложений, отложенных на "
        "проверку, ни одного слова нет в словаре обучающей части: вектор пустой, и модель ставит "
        "всем таким предложениям один и тот же ответ. Символьные n-граммы находят общие куски почти "
        "у любых двух предложений — "
        + ("пустых векторов у них нет." if empty_characters == 0 else f"пустых векторов у них {empty_characters}.")
    )
    if characters.test[-1] > characters.test[-2]:
        st.info(
            "**Кривая на правом краю ещё растёт** — "
            f"с {characters.test[-2]:.3f} до {characters.test[-1]:.3f} на последнем шаге. "
            "Модель не насытилась данными: главный способ поднять качество — размечать дальше, "
            "а не менять модель. Пунктир сверху — точность на обучающих сегментах; разрыв "
            "между ним и нижней линией — переобучение, и с 160 сегментами он не закрывается."
        )

    st.subheader("Три средства из раздела 13")
    remedies = load_remedies()
    newsgroups = load_newsgroups_curve()
    growth = newsgroups["train_size"].iloc[-1] / newsgroups["train_size"].iloc[0]
    table = pd.DataFrame(
        {
            "средство": [REMEDY_LABELS[key][0] for key in remedies["remedy"]],
            "было": remedies["before"],
            "стало": remedies["after"],
            "прирост": [f"{gain:+.3f}" for gain in remedies["gain"]],
            "вывод": [REMEDY_LABELS[key][1] for key in remedies["remedy"]],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Подбор перебрал 48 комбинаций параметров и нашёл ровно исходную. Склейка признаков "
        "добавила шум. Третья строка — тот же метод на корпусе 20 Newsgroups при росте выборки "
        f"в {growth:.0f} раз: своего корпуса больше нет, и лабораторная берёт чужой."
    )
    st.plotly_chart(newsgroups_figure(newsgroups, _curve(CHARACTERS, LOGISTIC)), use_container_width=True)
    st.caption(
        "Наша кривая выше зелёной при сопоставимом объёме: наши четыре класса различаются резче, "
        "чем темы новостных групп. Но форма у кривых одна: пока данных мало, разметка продолжает "
        "поднимать качество — а подбор параметров и склейка признаков на нашем корпусе не дали ничего."
    )


def _render_acceptance() -> None:
    annotations = _annotations()
    median = _median_bleu()
    threshold = st.slider(
        "Порог приёмки по BLEU",
        0.0,
        1.0,
        float(round(median, 2)),
        0.01,
        help="Сегмент принимается без проверки, если его BLEU не ниже порога. По умолчанию — медиана корпуса.",
    )

    errors = acceptance_errors(annotations, threshold)
    columns = st.columns(3)
    columns[0].metric("Принято с критической ошибкой", errors[MISSED_CRITICAL])
    columns[1].metric("Отклонено без ошибок", errors[REJECTED_CORRECT])
    columns[2].metric("Принято всего", f"{errors['accepted']} из {len(annotations)}")

    sweep = acceptance_sweep(annotations, THRESHOLD_GRID)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(mqm_figure(annotations, threshold, median), use_container_width=True)
    with right:
        st.plotly_chart(acceptance_figure(sweep, threshold), use_container_width=True)

    if clean_thresholds(sweep).empty:
        safe = sweep[sweep["missed_critical"] == 0].iloc[0]
        critical = annotations[annotations["severity"] == CRITICAL].nlargest(1, "bleu").iloc[0]
        correct = annotations[annotations["severity"] == NONE].nsmallest(1, "bleu").iloc[0]
        accepted = int(safe["accepted"])
        left = (
            f"не принимается ни один из {len(annotations)}"
            if accepted == 0
            else f"из {len(annotations)} принимается только {accepted}"
        )
        st.warning(
            "**Порога, при котором приёмка по BLEU не ошибается, на этих сегментах нет.** "
            "Чтобы не пропустить ни одной критической ошибки, порог надо поднять выше BLEU самой "
            f"«похожей» из них — {critical['bleu']:.3f} у {critical['id']}. Тогда {left}: "
            "приёмка по метрике превращается в проверку всего подряд человеком. А безупречный перевод "
            f"{correct['id']} получает BLEU {correct['bleu']:.3f} и отклоняется почти при любом пороге."
        )

    correlation = rank_correlation(annotations["bleu"], penalties(annotations["severity"]))
    st.caption(
        f"Ранговая корреляция BLEU и штрафа MQM на этих десяти сегментах — {correlation:+.2f}: "
        "связи нет ни в какую сторону. Метрика мерит совпадение формы с эталоном, штраф — "
        "цену ошибки для пользователя. Это разные величины."
    )

    outcome = acceptance(annotations, threshold)
    flags = lab_flags(annotations, median)
    display = pd.DataFrame(
        {
            "id": annotations["id"],
            "тип": annotations["type"],
            "перевод": annotations["ru_mt"],
            "ошибка": annotations["category"],
            "серьёзность": annotations["severity"],
            "штраф": penalties(annotations["severity"]).astype(int),
            "BLEU": annotations["bleu"].round(3),
            "при пороге": outcome,
            "раздел 7": flags,
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Столбец «раздел 7» — правило самой работы: критическая ошибка при BLEU выше медианы "
        f"({median:.3f}) и верный перевод при BLEU ниже неё. Сработало на сегментах: "
        f"{', '.join(annotations.loc[flags != '', 'id'])}."
    )

    with st.expander("Веса MQM"):
        st.dataframe(
            pd.DataFrame(
                {
                    "серьёзность": list(SEVERITIES),
                    "штраф": [SEVERITY_WEIGHTS[severity] for severity in SEVERITIES],
                    "когда ставить": [SEVERITY_WHEN[severity] for severity in SEVERITIES],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Одна критическая ошибка весит как двадцать пять незначительных. Веса отражают "
            "стоимость последствий, а не частоту: лучше сто опечаток, чем одна кнопка, которая "
            "удаляет не тот файл."
        )


def _render_agreement() -> None:
    annotations = _annotations()
    st.markdown(
        "**Пример из работы.** Десять сегментов раздела 6, категории ошибок у двух разметчиков. "
        "Первый — это разметка самой работы, второй — её ячейка про согласие. Поменяйте ответы "
        "второго и посмотрите, что происходит с каппой."
    )

    second: list[str] = []
    columns = st.columns(2)
    # Filled top to bottom, not alternately, so that on a narrow screen, where
    # the columns stack, the segments keep the lab's order.
    half = (len(annotations) + 1) // 2
    for position, row in enumerate(annotations.itertuples()):
        with columns[0 if position < half else 1]:
            second.append(
                st.selectbox(
                    f"{row.id} · {_short(row.ru_mt)}",
                    CATEGORIES,
                    index=CATEGORIES.index(row.category_second),
                    help=f"Первый разметчик: {row.category}. {row.comment}",
                )
            )
    first = list(annotations["category"])
    stats = agreement(first, second)

    metrics = st.columns(3)
    metrics[0].metric("Совпали ответы", f"{stats.observed:.0%}")
    metrics[1].metric("Совпали бы случайно", f"{stats.expected:.0%}")
    metrics[2].metric("Каппа Коэна", "—" if np.isnan(stats.kappa) else f"{stats.kappa:.3f}")
    st.caption(f"Согласие: {kappa_reading(stats.kappa)}.")

    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(matrix_figure(agreement_matrix(first, second)), use_container_width=True)
    with right:
        disputed = [
            {"id": row.id, "перевод": row.ru_mt, "первый": row.category, "второй": label, "разбор": row.comment}
            for row, label in zip(annotations.itertuples(), second, strict=True)
            if row.category != label
        ]
        if disputed:
            st.dataframe(pd.DataFrame(disputed), use_container_width=True, hide_index=True)
        st.caption(
            "Расхождение содержательное, а не небрежность. «account» → «счёт» — ошибка точности "
            "(смысл другой) или терминологии (не тот термин)? Оба ответа защитимы, пока "
            "руководство для разметчиков не решило иначе. Написать это руководство — работа "
            "лингвиста, и задание 8 Л.р. № 3 — его заготовка."
        )

    st.divider()
    st.markdown(
        "**Почему не просто процент совпадений.** Модельный пример: у двух разметчиков одинаковые "
        "привычки, и на части сегментов они видят одно и то же, а на остальных отвечают наугад."
    )
    controls = st.columns(2)
    with controls[0]:
        share = st.slider(
            "Доля сегментов без ошибок",
            0.0,
            0.95,
            0.8,
            0.05,
            help="Остальное делится поровну между пятью категориями ошибок.",
        )
    with controls[1]:
        true_share = st.slider(
            "На какой доле сегментов разметчики согласны по существу",
            0.0,
            1.0,
            0.0,
            0.1,
            help="0 — совпадают только случайно; 1 — всегда видят одно и то же.",
        )

    raw = expected_raw_agreement(true_share, share)
    chance = chance_agreement(share)
    kappa = (raw - chance) / (1.0 - chance)
    metrics = st.columns(3)
    metrics[0].metric("Совпадение ответов", f"{raw:.0%}")
    metrics[1].metric("Из-за случайности", f"{chance:.0%}")
    metrics[2].metric("Каппа Коэна", f"{kappa:.2f}")

    sweep = chance_sweep(true_share, np.linspace(0.0, 0.95, 20))
    st.plotly_chart(chance_figure(sweep, share), use_container_width=True)
    st.caption(
        f"При {share:.0%} сегментов без ошибок два разметчика, которые вообще не согласны, "
        f"совпадают в {chance:.0%} случаев — просто потому, что оба чаще всего ставят «нет». "
        "Процент совпадений растёт вместе с этой долей; каппа вычитает случайные совпадения "
        "и остаётся на уровне настоящего согласия. Поэтому в отчёте приводят каппу."
    )


def main():
    st.set_page_config(page_title="Данные и разметка", page_icon="🏷️", layout="wide")

    st.title("Данные и разметка: во что упирается качество")
    st.caption(
        "Когда модель и признаки уже выбраны разумно, их донастройка даёт единицы процентов, "
        "а то и ноль; новые данные — десятки. "
        "Но данные — это разметка, а у разметки есть и цена, и надёжность. Всё это измеримо."
    )

    _render_sidebar()

    tab_data, tab_acceptance, tab_agreement = st.tabs(["Данные или модель", "Приёмка по BLEU", "Согласие разметчиков"])
    with tab_data:
        _render_data_or_model()
    with tab_acceptance:
        _render_acceptance()
    with tab_agreement:
        _render_agreement()


if __name__ == "__main__":
    main()
