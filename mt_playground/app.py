"""Streamlit interface for the machine-translation metrics playground."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mt_playground.checks import CHECK_EXPLANATIONS, NEG_RU, NEG_RU_MORPHOLOGY, formal_checks
from mt_playground.corpus import (
    CONTENT_TYPES,
    blind_spots,
    coverage_table,
    load_corpus,
    segment_label,
)
from mt_playground.metrics import bleu, chrf, ter
from mt_playground.mutations import MUTATIONS
from mt_playground.plotting import (
    blind_spot_figure,
    coverage_figure,
    distribution_figure,
    metric_comparison_figure,
)


def _score(hypothesis: str, reference: str) -> dict[str, float]:
    return {
        "BLEU": bleu(hypothesis, reference),
        "chrF": chrf(hypothesis, reference),
        "TER": ter(hypothesis, reference),
    }


def _set_hypothesis(text: str, action: str | None) -> None:
    """Store new text and make the input field read it back.

    The source of truth is ``hypothesis_text``, not the widget key: Streamlit
    discards a widget's state when the widget does not render, and the
    ``st.rerun()`` after a button press never reaches the input — the text used
    to vanish silently. The field's key carries the revision number, so the
    widget re-reads ``value``.
    """
    st.session_state.hypothesis_text = text
    st.session_state.editor_revision = st.session_state.get("editor_revision", 0) + 1
    st.session_state.last_action = action


def _ensure_segment_state(segment: pd.Series) -> None:
    """Reset the editable text when the segment changes."""
    if st.session_state.get("segment_id") != segment["id"]:
        st.session_state.segment_id = segment["id"]
        _set_hypothesis(segment["ru_mt"], None)


def _render_sidebar(corpus: pd.DataFrame):
    with st.sidebar:
        st.header("Сегмент")
        chosen_types = st.multiselect(
            "Тип контента",
            CONTENT_TYPES,
            default=list(CONTENT_TYPES),
            help="Корпус локализации: по 40 сегментов каждого типа.",
        )
        visible = corpus[corpus["type"].isin(chosen_types)] if chosen_types else corpus

        only_flagged = st.checkbox(
            "Только те, где сработали проверки",
            value=False,
            help="Восемь сегментов из 160 — реальные поломки в выходе модели.",
        )
        if only_flagged:
            visible = visible[visible["есть_замечания"]]

        if visible.empty:
            st.warning("Под фильтр не попал ни один сегмент.")
            visible = corpus

        labels = {segment_label(row): row["id"] for _, row in visible.iterrows()}
        chosen_label = st.selectbox("Какой сегмент разбираем", list(labels))
        segment_id = labels[chosen_label]

        st.header("Пороги приёмки")
        bleu_threshold = st.slider(
            "Порог по BLEU",
            0.0,
            1.0,
            0.45,
            0.01,
            help="Сегменты ниже порога уходят на ручную проверку.",
        )
        semantic_threshold = st.slider(
            "Порог по семантической близости",
            0.0,
            1.0,
            0.70,
            0.01,
        )

        st.header("Правило отрицания")
        fix_morphology = st.toggle(
            "Учитывать русскую морфологию",
            value=False,
            help=(
                "Правило из ЛР № 3 ищет отрицание списком словоформ и потому "
                "даёт две ложные тревоги на этом корпусе. Включите, чтобы "
                "увидеть, как они исчезают."
            ),
        )

        st.caption(
            "Корпус и эталонные переводы — учебный материал курса. Столбец "
            "`ru_mt` — настоящий выход модели Helsinki-NLP/opus-mt-en-ru."
        )
        st.divider()
        st.caption(
            "Другие playground'ы: [градиентный спуск](../) · "
            "[языковая модель](../lm/) · [векторизация](../vec/) · "
            "[память переводов](../tm/)"
        )

    return segment_id, bleu_threshold, semantic_threshold, fix_morphology


def _render_editor(segment: pd.Series) -> None:
    st.subheader("Перевод, который проверяем")

    st.markdown("**Оригинал**")
    st.code(segment["en"], language="text")
    st.markdown("**Эталон**")
    st.code(segment["ru_ref"], language="text")

    st.markdown("**Проверяемый перевод** — правьте его и смотрите на метрики справа")

    st.caption("Сломать одним кликом:")
    columns = st.columns(len(MUTATIONS))
    for column, (label, mutate, hint) in zip(columns, MUTATIONS, strict=True):
        if column.button(label, help=hint, use_container_width=True):
            result = mutate(st.session_state.hypothesis_text)
            if result is None:
                st.session_state.last_action = f"«{label}» здесь неприменимо"
            else:
                mutated, description = result
                _set_hypothesis(mutated, description)
            st.rerun()

    restore_mt, restore_ref = st.columns(2)
    if restore_mt.button("Вернуть выход модели", use_container_width=True):
        _set_hypothesis(segment["ru_mt"], None)
        st.rerun()
    if restore_ref.button("Подставить эталон", use_container_width=True):
        _set_hypothesis(segment["ru_ref"], "подставлен эталон — идеальный случай")
        st.rerun()

    edited = st.text_area(
        "Проверяемый перевод",
        value=st.session_state.hypothesis_text,
        key=f"hypothesis_editor_{st.session_state.editor_revision}",
        height=110,
        label_visibility="collapsed",
    )
    st.session_state.hypothesis_text = edited

    if st.session_state.get("last_action"):
        st.caption(f"Последняя правка: {st.session_state.last_action}")


def _render_verdict(
    segment: pd.Series,
    current: dict[str, float],
    baseline: dict[str, float],
    negation_rule,
    bleu_threshold: float,
) -> None:
    hypothesis = st.session_state.hypothesis_text
    fired = formal_checks(segment["en"], hypothesis, negation_rule)

    st.subheader("Оценка")
    columns = st.columns(3)
    for column, name in zip(columns, ["BLEU", "chrF", "TER"], strict=True):
        delta = current[name] - baseline[name]
        # Lower is better for TER, so a drop is what should light up green.
        colour = "inverse" if name == "TER" else "normal"
        column.metric(
            name,
            f"{current[name]:.3f}",
            delta=f"{delta:+.3f}" if abs(delta) > 1e-9 else None,
            delta_color=colour,
        )

    st.markdown("**Формальные проверки**")
    if fired:
        for name in fired:
            st.error(f"**{name}** — {CHECK_EXPLANATIONS[name]}")
    else:
        st.success("Все проверки молчат.")

    # Comparing against the model output will not do: on some segments it is
    # broken itself, and then "it got worse" never happens. Judge the current
    # state against the threshold set in the sidebar instead.
    if fired and current["BLEU"] >= bleu_threshold:
        st.warning(
            f"**Вот оно.** Проверки сработали ({', '.join(fired)}), "
            f"а BLEU {current['BLEU']:.3f} — не ниже порога {bleu_threshold:.2f}. "
            "Приёмка по одному числу это пропустит."
        )
    elif not fired and current["BLEU"] < bleu_threshold:
        st.info(
            f"**И наоборот.** BLEU {current['BLEU']:.3f} ниже порога "
            f"{bleu_threshold:.2f}, но ни одна проверка не сработала: метрика "
            "наказала перевод за форму, а не за смысл."
        )

    st.markdown("**Семантическая близость**")
    if hypothesis.strip() == str(segment["ru_mt"]).strip():
        st.metric("косинус к эталону", f"{segment['semantic']:.3f}")
    else:
        st.metric("косинус к эталону", "—")
        st.caption(
            "Считается многоязычным энкодером заранее: в браузер он не "
            "помещается, поэтому для правленого текста значение недоступно. "
            "Для исходного выхода модели — есть."
        )


def _render_corpus_tabs(
    corpus: pd.DataFrame,
    segment_id: str,
    bleu_threshold: float,
    semantic_threshold: float,
    current: dict[str, float],
) -> None:
    tab_blind, tab_cost, tab_spread, tab_data = st.tabs(
        ["Слепая зона", "Цена тревоги", "Распределения", "Корпус"]
    )

    with tab_blind:
        st.plotly_chart(
            blind_spot_figure(corpus, bleu_threshold, segment_id),
            use_container_width=True,
        )
        missed = blind_spots(corpus, bleu_threshold)
        if missed.empty:
            st.success(f"При пороге BLEU {bleu_threshold:.2f} ни одна поломка не осталась незамеченной.")
        else:
            st.error(
                f"Поломок выше порога BLEU {bleu_threshold:.2f}: {len(missed)}. "
                "Метрика их пропустила, формальная проверка — нет."
            )
            st.dataframe(missed, use_container_width=True, hide_index=True)

    with tab_cost:
        table = coverage_table(corpus, bleu_threshold, semantic_threshold)
        st.plotly_chart(coverage_figure(table), use_container_width=True)
        st.caption(
            "Порог по метрике отправляет на проверку половину корпуса, "
            "формальные проверки — единицы процентов. Обе цифры реальные: "
            "это стоимость приёмки в человеко-часах."
        )

    with tab_spread:
        metric_name = st.radio("Метрика", ["BLEU", "chrF", "TER"], horizontal=True)
        st.plotly_chart(
            distribution_figure(corpus, metric_name, current[metric_name]),
            use_container_width=True,
        )

    with tab_data:
        st.dataframe(
            corpus[["id", "type", "en", "ru_ref", "ru_mt", "BLEU", "chrF", "TER", "semantic"]],
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(
        page_title="Метрики машинного перевода",
        page_icon="microscope",
        layout="wide",
    )

    st.title("Метрики машинного перевода: что они видят и что пропускают")
    st.caption(
        "Правьте перевод и следите за метриками. Задача — найти правку, которая "
        "ломает строку, не задев BLEU, и правку, которая роняет BLEU, не задев смысл."
    )

    baseline_corpus = load_corpus()
    segment_id, bleu_threshold, semantic_threshold, fix_morphology = _render_sidebar(baseline_corpus)

    negation_rule = NEG_RU_MORPHOLOGY if fix_morphology else NEG_RU
    corpus = load_corpus(negation_rule) if fix_morphology else baseline_corpus
    if fix_morphology:
        removed = int(baseline_corpus["есть_замечания"].sum() - corpus["есть_замечания"].sum())
        if removed:
            st.info(
                f"Правило с учётом морфологии снимает {removed} ложные тревоги: "
                "«Невозможно подключиться» и «Результаты отсутствуют» — это "
                "отрицания, которых исходный список словоформ не видит."
            )

    segment = corpus[corpus["id"] == segment_id].iloc[0]
    _ensure_segment_state(segment)

    left, right = st.columns([3, 2])
    with left:
        _render_editor(segment)
    current = _score(st.session_state.hypothesis_text, segment["ru_ref"])
    baseline = _score(segment["ru_mt"], segment["ru_ref"])
    with right:
        _render_verdict(segment, current, baseline, negation_rule, bleu_threshold)

    st.plotly_chart(metric_comparison_figure(current, baseline), use_container_width=True)

    _render_corpus_tabs(corpus, segment_id, bleu_threshold, semantic_threshold, current)


if __name__ == "__main__":
    main()
