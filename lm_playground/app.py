"""Streamlit interface for the language-model playground."""

from __future__ import annotations

import html
from functools import lru_cache

import streamlit as st

from lm_playground.corpus import CONTENT_TYPES, corpus_statistics, load_texts, split
from lm_playground.evaluation import (
    copied_fraction,
    copied_mask,
    longest_copied_span,
    perplexity,
    sweep_orders,
)
from lm_playground.generation import generate, split_segments
from lm_playground.model import CharNgramLM
from lm_playground.plotting import distribution_figure, overfitting_figure
from lm_playground.sampling import prepare

MAX_ORDER = 8
COPY_ALARM = 0.5  # copied share past which the text can no longer be called generated

# The train/held-out split is fixed and independent of the generation seed.
# Otherwise the "random seed" slider would shift the overfitting curve as well,
# and the student would see noise where there is a pattern to see.
SPLIT_SEED = 0
SWEEP_SEED = 0


@lru_cache(maxsize=32)
def _texts(content_types: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(load_texts(content_types))


@lru_cache(maxsize=32)
def _split(content_types: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train, heldout = split(list(_texts(content_types)), seed=SPLIT_SEED)
    return tuple(train), tuple(heldout)


@lru_cache(maxsize=32)
def _model(content_types: tuple[str, ...], order: int) -> CharNgramLM:
    train, _ = _split(content_types)
    return CharNgramLM(order).fit(list(train))


@lru_cache(maxsize=32)
def _sweep(content_types: tuple[str, ...]) -> tuple:
    train, heldout = _split(content_types)
    return tuple(sweep_orders(list(train), list(heldout), range(1, MAX_ORDER + 1), seed=SWEEP_SEED))


def _render_sidebar():
    with st.sidebar:
        st.header("Обучающий текст")
        chosen_types = st.multiselect(
            "Тип контента",
            CONTENT_TYPES,
            default=list(CONTENT_TYPES),
            help="Модель учится только на выбранных типах. Оставьте один — станет заметно, что стиль выучивается.",
        )

        st.header("Модель")
        order = st.slider(
            "Порядок n — сколько символов контекста",
            1,
            MAX_ORDER,
            3,
            1,
            help="Сколько предыдущих символов модель смотрит, выбирая следующий.",
        )

        st.header("Сэмплирование")
        temperature = st.slider(
            "Температура",
            0.1,
            2.0,
            1.0,
            0.05,
            help="Ниже 1 — предсказуемее и однообразнее, выше 1 — разнообразнее и бессвязнее.",
        )
        top_k = st.slider(
            "top-k (0 — выключено)",
            0,
            40,
            0,
            1,
            help="Оставить только k самых вероятных символов.",
        )
        top_p = st.slider(
            "top-p",
            0.1,
            1.0,
            1.0,
            0.05,
            help="Оставить минимальный набор символов с суммарной вероятностью p.",
        )

        st.header("Генерация")
        length = st.slider("Длина, символов", 60, 600, 240, 20)
        prompt = st.text_input("Затравка", value="", help="С чего начать. Можно оставить пустым.")
        seed = st.number_input(
            "Случайное зерно",
            min_value=0,
            max_value=9999,
            value=3,
            step=1,
            help="Меняет только жребий при генерации. Разбиение на обучение и отложенные закреплено.",
        )

        st.divider()
        st.caption(
            "Другие playground'ы: [градиентный спуск](../) · "
            "[метрики перевода](../mt/) · [векторизация](../vec/) · "
            "[память переводов](../tm/) · [данные и разметка](../labels/)"
        )

    return {
        "content_types": tuple(chosen_types) if chosen_types else CONTENT_TYPES,
        "order": int(order),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "length": int(length),
        "prompt": prompt,
        "seed": int(seed),
    }


def _highlight(generated: str, mask: list[bool]) -> str:
    """Build HTML with the verbatim-copied runs highlighted."""
    pieces: list[str] = []
    index = 0
    while index < len(generated):
        flagged = mask[index]
        start = index
        while index < len(generated) and mask[index] == flagged:
            index += 1
        chunk = html.escape(generated[start:index]).replace("\n", "<br>")
        pieces.append(f"<mark>{chunk}</mark>" if flagged else chunk)
    body = "".join(pieces)
    return (
        '<div style="line-height:1.7; font-size:0.95rem; white-space:pre-wrap; '
        'word-break:break-word">' + body + "</div>"
    )


def _render_generation(settings: dict, model: CharNgramLM, source: str) -> None:
    generation = generate(
        model,
        length=settings["length"],
        prompt=settings["prompt"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        top_p=settings["top_p"],
        seed=settings["seed"],
        keep_steps=1,
    )

    mask = copied_mask(generation.text, source)
    st.subheader("Что породила модель")
    st.caption("Подсвечено — куски длиной от 10 символов, дословно совпадающие с обучающим текстом.")
    st.markdown(_highlight(generation.text, mask), unsafe_allow_html=True)

    segments = split_segments(generation.text)
    if segments:
        with st.expander(f"Разбить на сегменты ({len(segments)})", expanded=False):
            for segment in segments:
                st.write(f"— {segment}")

    st.session_state.generation = generation
    st.session_state.copied_mask = mask


def _render_verdict(settings: dict, model: CharNgramLM, train: list[str], heldout: list[str], source: str) -> None:
    generation = st.session_state.generation
    share = copied_fraction(generation.text, source)
    longest = longest_copied_span(generation.text, source)

    st.subheader("Измерения")
    left, right = st.columns(2)
    left.metric(
        "Перплексия на отложенных",
        f"{perplexity(model, heldout):.2f}",
        help="Между сколькими символами модель колеблется на текстах, которых не видела.",
    )
    right.metric("Перплексия на обучении", f"{perplexity(model, train):.2f}")

    left_two, right_two = st.columns(2)
    left_two.metric("Списано дословно", f"{share * 100:.1f} %")
    right_two.metric("Длина совпадения", f"{len(longest)} симв.")

    if share >= COPY_ALARM:
        st.warning(
            f"**Осторожно с выводом.** {share * 100:.0f} % текста — дословные куски обучающего "
            "корпуса. Модель выглядит складной не потому, что научилась языку, "
            "а потому, что запомнила его."
        )
    elif share < 0.05 and settings["order"] <= 2:
        st.info(
            "Списывать ещё нечего: контекста в один-два символа не хватает даже "
            "на слово. Текст оригинален и бессмыслен одновременно."
        )

    if longest:
        st.caption("Самое длинное дословное совпадение:")
        st.code(longest, language="text")

    support = generation.steps[0].support if generation.steps else 0
    if support == 0 and settings["prompt"]:
        st.caption(
            "Для такой затравки контекст в обучении не встречался — модель "
            "откатилась на более короткий."
        )


def _render_tabs(settings: dict, model: CharNgramLM) -> None:
    tab_curve, tab_next, tab_corpus = st.tabs(
        ["Кривая переобучения", "Следующий символ", "Корпус"]
    )

    with tab_curve:
        results = _sweep(settings["content_types"])
        st.plotly_chart(
            overfitting_figure(list(results), settings["order"]),
            use_container_width=True,
        )
        best = min(results, key=lambda result: result.heldout_perplexity)
        st.info(
            f"Лучший компромисс — **n = {best.order}**: перплексия на отложенных "
            f"{best.heldout_perplexity:.2f}, списано {best.copied_fraction * 100:.1f} %. "
            "Дальше текст читается лучше, а модель всё больше пересказывает корпус наизусть."
        )

    with tab_next:
        context = (settings["prompt"] or "")[-model.order :]
        raw = model.distribution(context)
        prepared = prepare(
            raw,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
        )
        st.plotly_chart(distribution_figure(raw, prepared), use_container_width=True)
        survivors = sum(1 for value in prepared.values() if value > 0)
        st.caption(
            f"После температуры {settings['temperature']:.2f}, top-k {settings['top_k']} "
            f"и top-p {settings['top_p']:.2f} в розыгрыше осталось {survivors} символов "
            f"из {len(raw)}."
        )

    with tab_corpus:
        texts = list(_texts(settings["content_types"]))
        statistics = corpus_statistics(texts)
        columns = st.columns(3)
        columns[0].metric("Сегментов", statistics["segments"])
        columns[1].metric("Символов", statistics["characters"])
        columns[2].metric("Алфавит", statistics["alphabet"])
        st.caption(
            "Корпус учебный и маленький, и это часть задачи: на семи тысячах "
            "символов запоминание начинается рано и его хорошо видно."
        )
        st.dataframe({"текст": texts}, use_container_width=True, height=280)


def main():
    st.set_page_config(
        page_title="Языковая модель и температура",
        page_icon="abc",
        layout="wide",
    )

    st.title("Языковая модель: складно или просто списано?")
    st.caption(
        "Символьная n-граммная модель учится на корпусе локализации прямо в браузере. "
        "Крутите порядок модели и температуру и смотрите, что происходит с текстом — "
        "и с долей дословно скопированного."
    )

    settings = _render_sidebar()
    train, heldout = _split(settings["content_types"])
    source = "\n".join(train)
    model = _model(settings["content_types"], settings["order"])

    left, right = st.columns([3, 2])
    with left:
        _render_generation(settings, model, source)
    with right:
        _render_verdict(settings, model, list(train), list(heldout), source)

    _render_tabs(settings, model)


if __name__ == "__main__":
    main()
