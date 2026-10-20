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
from playground_common.explain import how_to_read, quick_start, reference
from playground_common.links import other_playgrounds
from playground_common.wording import plural

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
            help="Модель учится только на выбранных типах. Если оставить один, станет видно, что она усваивает стиль.",
        )

        st.header("Модель")
        order = st.slider(
            "Сколько предыдущих символов видит модель",
            1,
            MAX_ORDER,
            3,
            1,
            help=(
                "Главная настройка страницы. Модель выбирает очередной символ, учитывая только "
                "n предыдущих. При n = 1 она помнит одну букву и порождает бессмыслицу; при n = 7 "
                "помнит почти слово и начинает дословно воспроизводить корпус. "
                "В учебниках это число называют порядком n-граммной модели."
            ),
        )

        st.header("Как выбирается очередной символ")
        st.caption(
            "Модель не называет один символ, а выдаёт вероятности всех символов сразу. "
            "Эти три настройки определяют, как по этим вероятностям выбирается символ."
        )
        temperature = st.slider(
            "Температура",
            0.1,
            2.0,
            1.0,
            0.05,
            help=(
                "Насколько модели позволено рисковать. Ниже 1 она тяготеет к самому вероятному "
                "символу, и текст предсказуем и однообразен. Выше 1 выбор смелее, а текст "
                "разнообразнее и бессвязнее. При 1 вероятности используются без изменений."
            ),
        )
        top_k = st.slider(
            "Оставить k самых вероятных вариантов (0 — не ограничивать)",
            0,
            40,
            0,
            1,
            help=(
                "Отсекает маловероятные варианты: в выборе участвуют только k самых вероятных символов. "
                "Защищает от редких нелепых символов, которые иначе изредка выпадают."
            ),
        )
        top_p = st.slider(
            "Оставить варианты, набирающие вероятность p",
            0.1,
            1.0,
            1.0,
            0.05,
            help=(
                "То же отсечение, но гибче: берутся самые вероятные символы, пока их "
                "суммарная вероятность не достигнет p. Там, где модель уверена, останется "
                "один-два варианта; где не уверена — много."
            ),
        )

        st.header("Генерация")
        length = st.slider(
            "Длина, символов", 60, 600, 240, 20,
            help="Сколько символов породить за один раз.",
        )
        prompt = st.text_input(
            "Затравка",
            value="",
            help="Начало текста, которое модель продолжит. Можно оставить пустым.",
        )
        seed = st.number_input(
            "Случайное зерно",
            min_value=0,
            max_value=9999,
            value=3,
            step=1,
            help=(
                "Задаёт начальное состояние генератора случайных чисел: при том же зерне результат "
                "повторяется, при другом получится другой текст с теми же настройками. На саму модель не влияет."
            ),
        )

        st.divider()
        st.caption(other_playgrounds("lm"))

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
    st.subheader("2. Что породила модель")
    st.caption("Подсвечены фрагменты длиной от 10 символов, дословно совпадающие с обучающим текстом.")
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

    st.subheader("3. Измерения")
    st.caption(
        "Первая пара чисел — перплексия: насколько модель «удивлена» текстом (меньше — увереннее). "
        "Вторая — сколько в порождённом тексте дословно переписано из обучающего."
    )
    left, right = st.columns(2)
    left.metric(
        "Перплексия на новых текстах",
        f"{perplexity(model, heldout):.2f}",
        help=(
            "Приближённо: из скольких символов модель в среднем выбирает на каждом шаге "
            "на текстах, которых она не видела. Чем меньше, тем лучше. Это честная оценка: "
            "именно так модель поведёт себя на новом материале."
        ),
    )
    right.metric(
        "Перплексия на том, чему училась",
        f"{perplexity(model, train):.2f}",
        help=(
            "То же число на обучающем тексте. Оно обычно лучше первого, поэтому важен разрыв: "
            "чем он больше, тем сильнее модель запомнила корпус вместо того, чтобы "
            "уловить закономерность."
        ),
    )

    left_two, right_two = st.columns(2)
    left_two.metric(
        "Списано дословно", f"{share * 100:.1f} %",
        help="Какая доля порождённого текста совпадает с обучающим фрагментами от 10 символов.",
    )
    right_two.metric(
        "Длина совпадения", f"{len(longest)} симв.",
        help="Самый длинный фрагмент, дословно переписанный из обучающего текста.",
    )

    if share >= COPY_ALARM:
        st.warning(
            f"**Осторожно с выводом.** {share * 100:.0f} % текста — дословные фрагменты обучающего "
            "корпуса. Текст выглядит складным не потому, что модель научилась языку, "
            "а потому, что запомнила его."
        )
    elif share < 0.05 and settings["order"] <= 2:
        st.info(
            "Списывать ещё нечего: контекста в один-два символа недостаточно даже "
            "на слово. Текст оригинален и бессмыслен одновременно."
        )

    if longest:
        st.caption("Самое длинное дословное совпадение:")
        st.code(longest, language="text")

    support = generation.steps[0].support if generation.steps else 0
    if support == 0 and settings["prompt"]:
        st.caption(
            "Такой контекст в обучающем тексте не встречался, поэтому модель "
            "перешла к более короткому."
        )


def _render_tabs(settings: dict, model: CharNgramLM) -> None:
    st.subheader("4. Подробнее")
    tab_curve, tab_next, tab_corpus = st.tabs(
        ["Где модель начинает списывать", "Выбор следующего символа", "Обучающий корпус"]
    )

    with tab_curve:
        st.caption(
            "Модель обучена при каждом n от 1 до максимума. Левая шкала — перплексия на обучающих и на "
            "отложенных текстах, правая — доля дословно списанного. Текст становится всё складнее, "
            "но за счёт того, что модель всё больше пересказывает корпус наизусть. "
            "Вертикальные линии — лучший n и ваша текущая настройка."
        )
        results = _sweep(settings["content_types"])
        st.plotly_chart(
            overfitting_figure(list(results), settings["order"]),
            use_container_width=True,
        )
        best = min(results, key=lambda result: result.heldout_perplexity)
        st.info(
            f"Лучший компромисс — **n = {best.order}**: перплексия на отложенных "
            f"{best.heldout_perplexity:.2f}, списано {best.copied_fraction * 100:.1f} %. "
            "При больших n текст читается лучше, но модель всё больше пересказывает корпус наизусть."
        )

    with tab_next:
        st.caption(
            "Один шаг генерации крупным планом: какие символы модель считала возможными "
            "продолжениями затравки и что от этого списка осталось после температуры и отсечения. "
            "Из оставшихся символов и делается случайный выбор."
        )
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
            f"и top-p {settings['top_p']:.2f} в розыгрыше символов: {survivors} из {len(raw)}."
        )

    with tab_corpus:
        texts = list(_texts(settings["content_types"]))
        statistics = corpus_statistics(texts)
        columns = st.columns(3)
        columns[0].metric("Сегментов", statistics["segments"])
        columns[1].metric("Символов", statistics["characters"])
        columns[2].metric("Алфавит", statistics["alphabet"])
        st.caption(
            "Корпус учебный и небольшой, и это часть задачи: на семи тысячах "
            "символов запоминание начинается рано и хорошо заметно."
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
        "Меняйте порядок модели и температуру и следите за текстом "
        "и за долей дословно скопированного."
    )

    quick_start(
        st,
        purpose=(
            "Языковая модель предсказывает следующий символ по предыдущим. Здесь простейшая такая модель "
            "учится прямо в браузере на корпусе курса."
        ),
        steps=(
            "Двигайте **«Сколько предыдущих символов видит модель»** от 3 до 8.",
            "Следите за подсветкой в тексте и за числом **«Списано дословно»**: при 8 списано всё.",
            "Измените **«Температура»**: текст станет предсказуемее или случайнее.",
        ),
        notice=(
            "текст становится складнее ровно настолько, насколько модель перестаёт сочинять и начинает "
            "цитировать."
        ),
    )
    how_to_read(
        st,
        what=(
            "**Языковая модель** лежит в основе ChatGPT и автодополнения в телефоне: "
            "она умеет только предсказывать следующий символ по предыдущим и делает "
            "это много раз подряд. Здесь модель простейшего вида обучается прямо в "
            "браузере за долю секунды, поэтому её устройство можно рассмотреть целиком."
        ),
        controls=(
            "Главная настройка слева — сколько предыдущих символов модель видит. Ниже — три "
            "настройки того, как она выбирает символ из своих вариантов. Текст порождается заново "
            "сразу после изменения; можно задать затравку и зерно."
        ),
        watch=(
            "Следите одновременно за тем, насколько текст складный и сколько в нём **списано** из "
            "обучающего корпуса (это подсвечено прямо в тексте). Увеличивая длину контекста, "
            "вы увидите главное: текст становится красивее ровно в той мере, в какой модель "
            "перестаёт сочинять и начинает цитировать."
        ),
        data=(
            "Тот же корпус локализации, что и на других страницах, около семи тысяч символов. "
            "Он намеренно небольшой: на большом корпусе запоминание тоже есть, но его "
            "трудно заметить."
        ),
    )

    settings = _render_sidebar()
    train, heldout = _split(settings["content_types"])
    source = "\n".join(train)
    model = _model(settings["content_types"], settings["order"])

    st.subheader("1. Как модель порождает текст")
    st.markdown(
        f"Модель прочитала обучающие тексты и для каждой цепочки из n = {settings['order']} "
        f"{plural(settings['order'], 'символа', 'символов', 'символов')} запомнила, какие символы и как часто "
        "шли следом. Чтобы породить текст, она берёт последние n символов, получает из таблицы "
        "вероятности следующего символа, выбирает один из них случайно и повторяет. **Температура** и "
        "**отсечение** (top-k, top-p) меняют эти вероятности перед выбором."
    )
    left, right = st.columns([3, 2])
    with left:
        _render_generation(settings, model, source)
    with right:
        _render_verdict(settings, model, list(train), list(heldout), source)

    _render_tabs(settings, model)

    reference(st, "ngram", "perplexity", "temperature", "top_k_p", "overfitting")


if __name__ == "__main__":
    main()
