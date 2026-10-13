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
FAILURES_LABEL = "Сколько раз подряд она даёт сбой"
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
    st.sidebar.caption("Эти настройки меняют только поведение одной задачи.")
    failing = st.sidebar.selectbox(
        FAILING_LABEL,
        [NOBODY, *task_names()],
        index=task_names().index("обучение") + 1,
        help="Любая задача конвейера может завершиться сбоем: не ответил сервер, закончилось место, "
        "пришли повреждённые данные. Выберите, какая задача даёт сбой здесь.",
    )
    failures = st.sidebar.select_slider(
        FAILURES_LABEL, FAILURE_OPTIONS, value="2",
        help="Сколько первых попыток заканчиваются неудачей. «Всегда» означает, что задача не проходит "
        "ни при каком числе попыток.",
    )

    st.sidebar.header("Планировщик")
    st.sidebar.caption("Эта настройка меняет только терпение планировщика.")
    attempts = st.sidebar.slider(
        ATTEMPTS_LABEL, 1, 5, 3,
        help=(
            "Сколько раз планировщик пробует запустить задачу после сбоя, прежде чем признать её проваленной. Повторы "
            "спасают от случайных сбоев и бессильны против устойчивых."
        ),
    )

    st.sidebar.header("Граф")
    st.sidebar.caption("Эти настройки меняют только рёбра графа.")
    drop_guard = st.sidebar.checkbox(
        GUARD_LABEL, value=False,
        help="Только это ребро не позволяет опубликовать модель, которую никто "
        "не оценил. Уберите его и посмотрите, что произойдёт при сбое обучения.",
    )
    add_cycle = st.sidebar.checkbox(
        CYCLE_LABEL, value=False,
        help="Создаёт замкнутый круг: публикация до выгрузки, выгрузка до публикации. "
        "Такой граф выполнить нельзя, и планировщик должен отвергнуть его до запуска.",
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
            "Сбой с восстановлением: «обучение» дважды завершается сбоем и проходит с третьей попытки, "
            "конвейер выполняется целиком. Повторы спасли запуск."
        )
    elif scenario == FAILING_RUN:
        st.info(
            "Сбой без восстановления: «обучение» не проходит ни с одной из двух попыток. "
            "Всё, что от него зависит, пропущено, и модель без оценки не опубликована."
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
        help="Задачи, выполненные до конца, сразу или с повторной попытки.",
    )
    middle.metric(
        "Пропущено",
        str(outcome.skipped),
        help="Задачи, которые даже не запускались: то, от чего они зависят, не выполнилось. "
        "Это не сбой, а защита: иначе в работу уйдёт незавершённый результат.",
    )
    right.metric(
        "Попыток всего",
        str(outcome.total_attempts),
        help="Сумма запусков по всем задачам. Если она больше числа задач, были повторы.",
    )

    published = outcome.status.get("публикация")
    if outcome.published_unverified():
        st.warning(
            "Модель опубликована без оценки. Обучение не удалось, оценка пропущена, а публикация "
            "выполнилась, потому что ребро «оценка → публикация» убрано и её ничто не удерживает. "
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
                "вместо того чтобы остановить весь конвейер при первом же сбое."
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
            "Показать, как конвейер обучения модели, записанный графом зависимостей между задачами, "
            "переносит сбои."
        ),
        steps=(
            "Начните с настроек по умолчанию: «обучение» дважды даёт сбой, а конвейер всё равно выполняется до конца.",
            "Выберите в **«Сколько раз подряд она даёт сбой»** вариант «всегда»: зависимые задачи будут пропущены.",
            "Включите флажок **«Убрать ребро „оценка → публикация“»**: модель будет опубликована без проверки.",
        ),
        notice="публикацию непроверенной модели останавливают зависимости в графе, а не чья-то внимательность.",
    )
    how_to_read(
        st,
        what=(
            "Обученная модель — это не один скрипт, а цепочка шагов: выгрузить данные, "
            "очистить их, собрать признаки, обучить модель, оценить и опубликовать её. Эту "
            "цепочку записывают графом зависимостей между задачами. Граф даёт три возможности, "
            "которых у обычного скрипта нет. Он сам выводит порядок запуска. Он повторяет задачу "
            "после сбоя, не останавливая всё остальное. И он не запускает то, что зависело от "
            "проваленного шага, поэтому неоценённая модель не уйдёт в работу. Здесь всё это можно "
            "нарушить и посмотреть, от чего именно защищает граф."
        ),
        controls=(
            "**Сбой** — какая задача даёт сбой и сколько раз подряд. **Планировщик** — "
            "сколько попыток он даёт каждой задаче. **Граф** — два флажка, меняющих "
            "сами зависимости: убрать защитное ребро «оценка → публикация» или добавить "
            "ребро, создающее замкнутый круг."
        ),
        watch=(
            "Начните с настроек по умолчанию: «обучение» даёт сбой в двух попытках из трёх, "
            "и конвейер всё равно выполняется до конца. Затем выберите «всегда»: обучение "
            "проваливается, всё, что от него зависело, помечено серым «пропущена», и "
            "публикации не происходит. Далее самое важное: при том же сбое уберите ребро "
            "«оценка → публикация». Модель будет опубликована без проверки — в точности так "
            "поступил бы скрипт без графа зависимостей. Наконец, включите второй "
            "флажок: планировщик вообще откажется запускать граф, потому что в нём появился "
            "замкнутый круг."
        ),
        data=(
            "Конвейер и планировщик учебные: они выполняются мгновенно и ничего не вычисляют, "
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
            "раньше, чем выгружены данные, а выгрузка нужна для публикации. Планировщик должен "
            "отвергнуть такой граф до запуска первой задачи, и он его отвергает."
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
        "Задачи в одном столбце не зависят друг от друга и могли бы выполняться одновременно."
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
            "уровня — тоже. Параллелизм здесь показан, но не используется.\n"
            "- Не ждёт между попытками: пауза в браузере ничего бы не показала.\n"
            "- Не воспроизводит Airflow: здесь нет расписания, тайм-аутов и состояния между запусками. "
            "Их дают настоящие оркестраторы.\n"
            "- Не допускает одновременных сбоев нескольких задач: каждая настройка должна менять одну причину."
        )


if __name__ == "__main__":
    main()
