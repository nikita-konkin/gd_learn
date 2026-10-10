"""Streamlit interface for the task-graph playground.

Two runs show the scheduler's whole point: one with a task that recovers on
the third attempt, one with a task that never recovers. This page lets the student pick
the task, the number of failures, the number of attempts and two edges of the
graph — and shows, in the journal and on the graph, what the scheduler does.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import streamlit as st

from dag_playground.plotting import graph_figure
from dag_playground.scheduler import (
    ALWAYS,
    CYCLE_EDGE,
    FAILING_RUN,
    GUARD_EDGE,
    RECOVERING_RUN,
    SKIPPED,
    SUCCESS,
    Outcome,
    Scenario,
    simulate,
    task_names,
)
from playground_common.explain import how_to_read, quick_start, reference
from playground_common.links import other_playgrounds
from playground_common.wording import plural

TITLE = "Граф задач и отказы"

FAILING_LABEL = "Какая задача сбоит"
FAILURES_LABEL = "Сколько раз подряд она падает"
ATTEMPTS_LABEL = "Попыток на задачу"
GUARD_LABEL = "Убрать ребро «оценка → публикация»"
CYCLE_LABEL = "Добавить ребро «публикация → выгрузка»"

NOBODY = "никакая"
FOREVER = "всегда"
FAILURE_OPTIONS = ["0", "1", "2", "3", "4", "5", FOREVER]


@lru_cache(maxsize=256)
def _simulate(scenario: Scenario) -> Outcome:
    return simulate(scenario)


def _arrow(edge: tuple[str, str]) -> str:
    return f"{edge[0]} → {edge[1]}".replace("_", " ")


def _controls() -> Scenario:
    """The sidebar. Each control changes one thing: where, how often, how patient, which edges."""
    st.sidebar.header("Сбой")
    st.sidebar.caption("Эти ручки меняют только поведение одной задачи.")
    failing = st.sidebar.selectbox(
        FAILING_LABEL,
        [NOBODY, *task_names()],
        index=task_names().index("обучение") + 1,
        help="Любая задача конвейера может упасть: не ответил сервер, кончилось место, "
        "пришли битые данные. Выберите, какая падает здесь.",
    )
    failures = st.sidebar.select_slider(
        FAILURES_LABEL, FAILURE_OPTIONS, value="2",
        help="Сколько первых попыток заканчиваются неудачей. «Всегда» — задача не проходит "
        "никогда, сколько бы попыток ей ни дали.",
    )

    st.sidebar.header("Планировщик")
    st.sidebar.caption("Эта ручка меняет только терпение планировщика.")
    attempts = st.sidebar.slider(
        ATTEMPTS_LABEL, 1, 5, 3,
        help=(
            "Сколько раз планировщик пробует запустить упавшую задачу, прежде чем признать её проваленной. Повторы "
            "спасают от случайных сбоев и бессильны против настоящих."
        ),
    )

    st.sidebar.header("Граф")
    st.sidebar.caption("Эти ручки меняют только рёбра графа.")
    drop_guard = st.sidebar.checkbox(
        GUARD_LABEL, value=False,
        help="Это ребро — единственное, что не даёт опубликовать модель, которую никто "
        "не оценил. Уберите его и посмотрите, что произойдёт при упавшем обучении.",
    )
    add_cycle = st.sidebar.checkbox(
        CYCLE_LABEL, value=False,
        help="Создаёт замкнутый круг: публикация до выгрузки, выгрузка до публикации. "
        "Такой граф выполнить нельзя, и планировщик обязан отвергнуть его до запуска.",
    )

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/dag"))

    return Scenario(
        failing_task=None if failing == NOBODY else failing,
        failures=ALWAYS if failures == FOREVER else int(failures),
        attempts=attempts,
        guard=not drop_guard,
        cycle=add_cycle,
    )


def _scenario_note(scenario: Scenario) -> None:
    if scenario == RECOVERING_RUN:
        st.info(
            "Сбой с восстановлением: «обучение» падает дважды и проходит с третьей попытки, "
            "конвейер выполняется целиком. Повторы спасли запуск."
        )
    elif scenario == FAILING_RUN:
        st.info(
            "Сбой без восстановления: «обучение» не проходит ни с одной из двух попыток. "
            "Всё, что от него зависит, пропущено, — и модель без оценки не опубликована."
        )


def _levels(outcome: Outcome) -> None:
    st.subheader("Порядок выполнения")
    lines = []
    for number, level in enumerate(outcome.levels, 1):
        tasks = ", ".join(task.replace("_", " ") for task in level)
        mark = " — одновременно" if len(level) > 1 else ""
        lines.append(f"{number}. {tasks}{mark}")
    st.markdown("\n".join(lines))


def _journal(outcome: Outcome) -> None:
    st.subheader("Журнал запуска")
    table = pd.DataFrame(
        {
            "задача": [entry.task.replace("_", " ") for entry in outcome.log],
            "итог": [entry.outcome for entry in outcome.log],
            "попыток": [entry.attempts for entry in outcome.log],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _verdict(outcome: Outcome, scenario: Scenario) -> None:
    total = len(outcome.status)
    left, middle, right = st.columns(3)
    left.metric(
        "Выполнено успешно",
        f"{outcome.succeeded} из {total}",
        help="Задачи, дошедшие до конца — сразу или с повторной попытки.",
    )
    middle.metric(
        "Пропущено",
        str(outcome.skipped),
        help="Задачи, которые даже не запускались: то, от чего они зависят, не выполнилось. "
        "Это не сбой, а защита — иначе в работу уйдёт недоделанный результат.",
    )
    right.metric(
        "Попыток всего",
        str(outcome.total_attempts),
        help="Сумма запусков по всем задачам. Больше числа задач — значит были повторы.",
    )

    published = outcome.status.get("публикация")
    if outcome.published_unverified():
        st.warning(
            "Модель опубликована без оценки. Обучение не удалось, оценка пропущена — а публикация "
            "выполнилась, потому что ребро «оценка → публикация» убрано и её ничто не держит. "
            "Скрипт без графа зависимостей поступил бы так же."
        )
    elif published == SKIPPED:
        st.success(
            "Публикация не выполнена: её зависимости не завершились успешно. Это сделал граф, "
            "а не проверка, которую кто-то не забыл написать."
        )
    elif published == SUCCESS and scenario.failing_task is not None and scenario.failures != 0:
        retried = [entry for entry in outcome.log if entry.status == SUCCESS and entry.attempts > 1]
        if retried:
            entry = retried[0]
            st.caption(
                f"«{entry.task.replace('_', ' ')}» прошла с {entry.attempts}-й попытки — "
                "вместо того чтобы уронить весь конвейер при первой же неурядице."
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Планировщик на сорока строках: топологический порядок, повторы при сбое "
        "и пропуск задач, чьи зависимости не выполнены."
    )

    quick_start(
        st,
        purpose=(
            "Конвейер обучения модели как граф задач: кто от кого зависит. Страница показывает, как граф "
            "переживает сбои."
        ),
        steps=(
            "По умолчанию «обучение» падает дважды, а конвейер всё равно доходит до конца.",
            "Поставьте **«Сколько раз подряд она падает»** — «всегда»: зависимые задачи будут пропущены.",
            "Включите флажок **Убрать ребро «оценка → публикация»**: модель опубликуется непроверенной.",
        ),
        notice="публикацию непроверенной модели останавливают зависимости в графе, а не чья-то внимательность.",
    )
    how_to_read(
        st,
        what=(
            "Обученная модель — это не один скрипт, а цепочка шагов: выгрузить данные, "
            "очистить, собрать признаки, обучить, оценить, опубликовать. Записывают эту "
            "цепочку графом: кто от кого зависит. Граф даёт три вещи, которых у обычного "
            "скрипта нет. Он сам выводит порядок запуска. Он повторяет упавшую задачу, "
            "не роняя всё остальное. И он не запускает то, что зависело от провалившегося "
            "шага, — поэтому неоценённая модель не уйдёт в работу. Здесь всё это можно "
            "сломать и посмотреть, что именно спасает граф."
        ),
        controls=(
            "**Сбой** — какая задача падает и сколько раз подряд. **Планировщик** — "
            "сколько попыток он даёт каждой задаче. **Граф** — два флажка, меняющих "
            "сами зависимости: убрать защитное ребро «оценка → публикация» или добавить "
            "ребро, создающее замкнутый круг."
        ),
        watch=(
            "Сначала настройки по умолчанию: «обучение» падает дважды из трёх попыток — "
            "и конвейер всё равно доходит до конца. Теперь поставьте «всегда»: обучение "
            "проваливается, и всё, что от него зависело, помечено серым «пропущено» — "
            "публикации не было. Дальше самое важное: при том же сбое уберите ребро "
            "«оценка → публикация». Модель опубликуется — непроверенная. Это в точности то, "
            "что сделал бы скрипт без графа зависимостей. И напоследок включите второй "
            "флажок: планировщик откажется запускать граф вообще, потому что в нём появился "
            "замкнутый круг."
        ),
        data=(
            "Конвейер и планировщик учебные, выполняются мгновенно и ничего не считают: "
            "задачи только сообщают об успехе или сбое. Логика порядка, повторов и пропусков "
            "при этом настоящая."
        ),
    )
    reference(st, "dag", "topological_order", "retry")

    scenario = _controls()
    outcome = _simulate(scenario)

    if outcome.cycle_error:
        st.error(
            f"{outcome.cycle_error}. Ребро «{_arrow(CYCLE_EDGE)}» требует опубликовать модель "
            "раньше, чем выгружены данные, а выгрузка нужна для публикации. Планировщик обязан "
            "отвергнуть такой граф до запуска первой задачи — и отвергает."
        )
        return

    _scenario_note(scenario)
    _verdict(outcome, scenario)

    edges = scenario.edges()
    st.plotly_chart(graph_figure(outcome.levels, edges, outcome.status), use_container_width=True)
    tasks, dependencies = len(outcome.status), len(edges)
    st.caption(
        f"{tasks} {plural(tasks, 'задача', 'задачи', 'задач')}, "
        f"{dependencies} {plural(dependencies, 'зависимость', 'зависимости', 'зависимостей')}. "
        "Синий — успех, оранжевый — провал, серый — пропущена. "
        "Задачи в одном столбце не зависят друг от друга и могли бы идти одновременно."
    )
    if not scenario.guard:
        st.caption(f"Ребро «{_arrow(GUARD_EDGE)}» убрано.")

    left, right = st.columns([1, 2])
    with left:
        _levels(outcome)
    with right:
        _journal(outcome)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не исполняет задачи одновременно: уровни выполняются по очереди, задачи внутри "
            "уровня — тоже. Параллельность здесь показана, а не использована.\n"
            "- Не ждёт между попытками: пауза в браузере ничего бы не показала.\n"
            "- Не повторяет Airflow: нет расписания, таймаутов, состояния между запусками. "
            "Их дают настоящие оркестраторы.\n"
            "- Сбоит только одна задача за раз — чтобы каждая ручка меняла одну причину."
        )


if __name__ == "__main__":
    main()
