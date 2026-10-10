from __future__ import annotations

import pandas as pd
import streamlit as st

from gd_playground.config import (
    CONVERGENCE_OPTIONS,
    DATA_DISTRIBUTIONS,
    DISTRIBUTION_NAMES,
    LEARNING_RATE_OPTIONS,
    MODEL_NAMES,
    MODEL_TYPES,
    OPTIMIZER_NAMES,
    OPTIMIZERS,
)
from gd_playground.data import clamp_batch_size, normalized_feature_text
from gd_playground.model import active_param_names, degree_for_model, equation_text
from gd_playground.plotting import animated_dashboard_figure, loss_figure, loss_surface_figure, model_figure
from gd_playground.state import (
    ensure_distinct_surface_params,
    init_state,
    params_vector,
    refresh_dataset,
    reset_all_state,
    sync_training_config,
)
from gd_playground.training import SETTLED_STEP, loss_function_options, optimization_loss_label
from gd_playground.workflow import (
    apply_animation_build,
    apply_run,
    apply_step,
    current_metrics,
    history_summary,
)
from playground_common.explain import how_to_read, quick_start, reference
from playground_common.links import other_playgrounds
from playground_common.wording import plural


def _render_actions(container):
    """The run buttons.

    Drawn into a container created first, so they land at the top of the
    sidebar. Before this they sat below fifteen sliders and had to be scrolled
    to every single time.
    """
    with container:
        st.subheader("Действия")
        col_a, col_b = st.columns(2)
        run_clicked = col_a.button(
            "Запустить",
            type="primary",
            use_container_width=True,
            help="Прогнать выбранное число итераций и показать итог.",
        )
        step_clicked = col_b.button(
            "Один шаг",
            use_container_width=True,
            help="Один шаг градиентного спуска — видно, куда сдвинулись параметры.",
        )

        col_c, col_d = st.columns(2)
        animation_clicked = col_c.button(
            "Показать анимацию",
            use_container_width=True,
            help="Собрать анимацию спуска с ползунком по итерациям.",
        )
        reset_clicked = col_d.button(
            "Сбросить",
            use_container_width=True,
            help="Вернуть все параметры и историю к значениям по умолчанию.",
        )

    return run_clicked, step_clicked, animation_clicked, reset_clicked


def _render_data_controls():
    with st.expander("Данные", expanded=True):
        st.session_state.distribution = st.selectbox(
            "Какую зависимость прячем в данных",
            DATA_DISTRIBUTIONS,
            index=DATA_DISTRIBUTIONS.index(st.session_state.distribution),
            format_func=lambda key: DISTRIBUTION_NAMES[key],
            help=(
                "Данные генерируются искусственно: берётся вот такая зависимость, к ней "
                "добавляется шум — и модель пытается её восстановить. Это удобно тем, что "
                "правильный ответ известен заранее и его видно на графике."
            ),
        )
        st.session_state.n_points = st.slider(
            "Сколько точек",
            5,
            250,
            int(st.session_state.n_points),
            1,
            help="Объём обучающей выборки. Чем меньше точек, тем сильнее модель зависит от случайности.",
        )
        st.session_state.noise = st.slider(
            "Насколько точки разбросаны (шум)",
            0.0,
            5.0,
            float(st.session_state.noise),
            0.1,
            help=(
                "Насколько сильно точки отклоняются от истинной зависимости. При нуле они "
                "лежат на ней идеально — так в жизни не бывает."
            ),
        )
        st.session_state.seed = st.number_input(
            "Случайное зерно",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.seed),
            step=1,
            help="Меняет жребий при генерации точек: другое зерно — другая выборка из той же зависимости.",
        )
        shuffle_now = st.button("Сгенерировать точки заново", use_container_width=True)
    return shuffle_now


def _render_model_controls():
    with st.expander("Модель", expanded=True):
        st.session_state.model_type = st.selectbox(
            "Какой кривой приближаем",
            MODEL_TYPES,
            index=MODEL_TYPES.index(st.session_state.model_type),
            format_func=lambda key: MODEL_NAMES[key],
            help=(
                "Чем сложнее кривая, тем больше у неё настраиваемых чисел и тем точнее она "
                "сядет на обучающие точки — вплоть до того, что начнёт повторять шум."
            ),
        )
        degree = degree_for_model(st.session_state.model_type)
        st.session_state.w0 = st.slider("w₀ — сдвиг по вертикали", -20.0, 30.0, float(st.session_state.w0), 0.01)
        st.session_state.w1 = st.slider("w₁ — наклон", -10.0, 10.0, float(st.session_state.w1), 0.01)

        if degree >= 2:
            st.session_state.w2 = st.slider("w₂ — при квадрате", -5.0, 5.0, float(st.session_state.w2), 0.01)
        else:
            st.session_state.w2 = 0.0

        if degree >= 3:
            st.session_state.w3 = st.slider("w₃ — при кубе", -2.0, 2.0, float(st.session_state.w3), 0.01)
        else:
            st.session_state.w3 = 0.0
    return degree


def _render_optimizer_controls():
    with st.expander("Оптимизатор", expanded=True):
        st.session_state.optimizer = st.selectbox(
            "Сколько точек смотреть перед одним шагом",
            OPTIMIZERS,
            index=OPTIMIZERS.index(st.session_state.optimizer),
            format_func=lambda key: OPTIMIZER_NAMES[key],
            help=(
                "Batch — все точки сразу: шаги точные, но каждый дорогой. SGD — одна случайная "
                "точка: шаги дёшевы и скачут. Мини-пакет — компромисс, его и используют на практике."
            ),
        )
        losses = loss_function_options()
        st.session_state.loss_function = st.selectbox(
            "Чем мерить ошибку (функция потерь)",
            losses,
            index=losses.index(st.session_state.loss_function),
            help=(
                "Число, которым измеряют расхождение предсказаний с данными. Именно его "
                "градиентный спуск и уменьшает. MSE возводит промахи в квадрат и потому "
                "болезненно реагирует на выбросы; MAE берёт модуль и к ним спокойнее."
            ),
        )
        st.session_state.learning_rate = st.select_slider(
            "Размер шага (learning rate)",
            options=LEARNING_RATE_OPTIONS,
            value=float(st.session_state.learning_rate),
            format_func=lambda value: f"{value:.0e}" if value < 0.001 else f"{value:g}",
            help=(
                "Насколько далеко сдвигать параметры за один шаг. Слишком маленький — спуск "
                "ползёт; слишком большой — перелетает минимум и улетает в бесконечность. "
                "Это самая капризная настройка во всём обучении нейросетей."
            ),
        )

        if st.session_state.optimizer == "Mini-batch SGD":
            st.session_state.batch_size = st.slider(
                "Размер пакета",
                2,
                max(2, int(st.session_state.n_points)),
                clamp_batch_size(st.session_state.batch_size, st.session_state.n_points),
                1,
                help="Сколько точек усредняется перед одним шагом. Меньше — шаги чаще и шумнее.",
            )
        elif st.session_state.optimizer == "SGD":
            st.session_state.batch_size = 1

        shuffle_each_epoch = st.checkbox("Перемешивать точки между проходами", value=True)
    return shuffle_each_epoch


def _render_run_length_controls():
    with st.expander("Длительность и сходимость", expanded=False):
        st.session_state.iterations_run = st.slider(
            "Шагов по кнопке «Запустить»",
            1,
            3000,
            int(st.session_state.iterations_run),
            1,
            help="Сколько шагов спуска делать за одно нажатие.",
        )
        st.session_state.iterations_animation = st.slider(
            "Шагов в анимации",
            10,
            600,
            int(st.session_state.iterations_animation),
            10,
            help="Сколько кадров собрать: каждый кадр — один шаг.",
        )
        st.session_state.convergence_tolerance = st.select_slider(
            "Когда считать, что процесс сошёлся",
            options=CONVERGENCE_OPTIONS,
            value=float(st.session_state.convergence_tolerance),
            format_func=lambda value: f"{value:.0e}",
            help=(
                "Если ошибка за шаг меняется меньше чем на эту величину, считаем, что спуск "
                "добрался до минимума и дальше идти некуда."
            ),
        )


def _render_sidebar_controls():
    with st.sidebar:
        # The container is created first and filled last: the buttons are
        # visible straight away but still know every other widget's value.
        actions = st.container()

        shuffle_now = _render_data_controls()
        degree = _render_model_controls()
        shuffle_each_epoch = _render_optimizer_controls()
        _render_run_length_controls()

        run_clicked, step_clicked, animation_clicked, reset_clicked = _render_actions(actions)

        st.divider()
        st.caption(other_playgrounds(""))

    return {
        "degree": degree,
        "shuffle_now": shuffle_now,
        "shuffle_each_epoch": shuffle_each_epoch,
        "run_clicked": run_clicked,
        "step_clicked": step_clicked,
        "animation_clicked": animation_clicked,
        "reset_clicked": reset_clicked,
    }


def _iterations(count: int) -> str:
    """«за 1 итерацию», «за 3 итерации», «за 100 итераций»."""
    return f"{count} {plural(count, 'итерацию', 'итерации', 'итераций')}"


def _barely_moved(summary) -> bool:
    """The run ended within one per cent of where it started."""
    return summary["last_optimization_loss"] >= 0.99 * summary["first_optimization_loss"]


def _render_training_status(summary):
    """The verdict on the run — above the table of numbers, not below it."""
    if summary is None:
        st.info(
            "Обучение ещё не запускалось. Нажмите слева **«Запустить»** — "
            "или **«Один шаг»**, если хотите посмотреть по одному шагу."
        )
        return

    if st.session_state.divergence_iteration is not None:
        st.error(
            f"Обучение разошлось на шаге {st.session_state.divergence_iteration}: "
            "ошибка начала расти и улетела в бесконечность. "
            f"Показан последний устойчивый шаг — {summary['max_iteration']}. "
            "Уменьшите размер шага."
        )
    elif summary["convergence_iteration"] is None and _barely_moved(summary):
        if summary["last_weight_step"] >= SETTLED_STEP:
            st.warning(
                "Ошибка не падает: шаг так велик, что каждое обновление перепрыгивает через дно "
                "и попадает на ту же высоту с другой стороны. Уменьшите размер шага."
            )
        else:
            st.warning(
                f"За {_iterations(summary['max_iteration'])} ошибка почти не изменилась: шаг так мал, "
                "что спуск едва ползёт. Увеличьте размер шага или дайте больше шагов."
            )
    elif summary["convergence_iteration"] is None:
        st.warning(
            f"За {_iterations(summary['max_iteration'])} ошибка всё ещё заметно падает — "
            "спуск не дошёл до дна. Дайте больше шагов или увеличьте их размер."
        )
    else:
        st.success(
            f"Сошлось примерно за {_iterations(summary['convergence_iteration'])}: "
            "дальше шаги почти не меняют ошибку."
        )


def _render_metrics_panel(data, degree: int):
    metrics, transform = current_metrics(st.session_state, data, degree)
    summary = history_summary(st.session_state.history, st.session_state.convergence_tolerance)

    st.subheader("Что происходит")
    _render_training_status(summary)

    # The three numbers people watch first. The other seven used to sit here
    # in one solid column and drown; they live under an expander now.
    loss_label = optimization_loss_label(st.session_state.loss_function)
    top_left, top_right = st.columns(2)
    top_left.metric(
        loss_label,
        f"{metrics['optimization_loss']:.6f}",
        help="Насколько предсказания расходятся с точками. Это и есть то число, которое спуск уменьшает.",
    )
    top_right.metric(
        "Крутизна склона",
        f"{metrics['grad_norm']:.6f}",
        help="Норма градиента. Большая — спуск ещё далеко от дна, около нуля — он уже пришёл.",
    )
    st.caption(
        f"Кривая: {MODEL_NAMES[st.session_state.model_type]} · "
        f"перед шагом смотрим {OPTIMIZER_NAMES[st.session_state.optimizer]} · "
        f"ошибку мерим как {st.session_state.loss_function}"
    )

    with st.expander("Все метрики", expanded=False):
        st.caption(
            "«На данных» — расхождение с теми точками, которые модель видит; в них есть шум. "
            "«По истинной зависимости» — расхождение с той кривой, из которой точки получены. "
            "В реальной задаче второе измерить нельзя: истинной кривой там никто не знает. "
            "Здесь она известна, и поэтому видно, где модель села на шум, а не на закономерность."
        )
        st.metric(
            optimization_loss_label(st.session_state.loss_function, "по истинной зависимости"),
            f"{metrics['true_optimization_loss']:.6f}",
        )
        st.metric("MSE на данных", f"{metrics['mse']:.6f}")
        st.metric("MSE по истинной зависимости", f"{metrics['true_mse']:.6f}")
        st.metric(
            "RMSE на данных",
            f"{metrics['rmse']:.6f}",
            help="Корень из MSE — средний промах в тех же единицах, что и сам ответ.",
        )
        st.metric(
            "MAE на данных",
            f"{metrics['mae']:.6f}",
            help="Средний промах по модулю. В отличие от MSE почти не раздувается выбросами.",
        )

    if summary is not None:
        with st.expander("Лучшее за историю обучения", expanded=False):
            st.caption(
                "Спуск не обязан улучшать результат на каждом шаге — особенно SGD, где "
                "каждый шаг считается по случайной точке. Здесь видно, какой результат "
                "был лучшим за весь прогон и на чём спуск в итоге остановился."
            )
            loss_caption = (
                f"Последний устойчивый {st.session_state.loss_function} на графике"
                if st.session_state.divergence_iteration is not None
                else f"{st.session_state.loss_function} на последнем шаге графика"
            )
            st.caption(f"{loss_caption}: {summary['last_optimization_loss']:.6f}")
            st.caption(
                f"Лучший {st.session_state.loss_function} на данных за прогон: "
                f"{summary['best_optimization_loss']:.6f}"
            )
            st.caption(
                f"Лучший {st.session_state.loss_function} по истинной зависимости за прогон: "
                f"{summary['best_true_optimization_loss']:.6f}"
            )
            st.caption(f"Лучший MSE на данных за прогон: {summary['best_observed_mse']:.6f}")
            st.caption(
                f"Лучший MSE по истинной зависимости за прогон: {summary['best_true_mse']:.6f}"
            )

    params = params_vector(st.session_state)
    st.caption("Текущее уравнение модели и то, как признак x приведён к единому масштабу:")
    st.code(
        f"{equation_text(params, degree, 'z')}\n{normalized_feature_text(transform)}",
        language="text",
    )

    if st.session_state.optimizer == "Batch GD":
        meaning = "Перед каждым шагом просматриваются все точки сразу: шаги точные, но дорогие."
    elif st.session_state.optimizer == "SGD":
        meaning = "Перед каждым шагом берётся одна случайная точка: шаги дешёвые, но скачут."
    else:
        size = int(st.session_state.batch_size)
        meaning = (
            f"Перед каждым шагом усредняется {size} "
            f"{plural(size, 'случайная точка', 'случайные точки', 'случайных точек')}."
        )
    st.caption(meaning)


def _render_visual_panel(data, degree: int):
    if st.session_state.animation_history:
        st.plotly_chart(
            animated_dashboard_figure(
                data,
                st.session_state.animation_history,
                degree,
                st.session_state.convergence_tolerance,
                st.session_state.divergence_iteration,
                st.session_state.loss_function,
            ),
            use_container_width=True,
            config={"displayModeBar": True},
        )
        st.info(
            "Нажмите **Play** внутри графика или протащите ползунок под ним: каждый кадр — "
            "один шаг спуска. Сверху видно, как кривая садится на точки, снизу — как при этом "
            "падает ошибка."
        )
        return

    st.plotly_chart(
        model_figure(data, params_vector(st.session_state), degree),
        use_container_width=True,
        config={"displayModeBar": True},
    )
    st.plotly_chart(
        loss_figure(
            st.session_state.history,
            data,
            degree,
            st.session_state.convergence_tolerance,
            st.session_state.divergence_iteration,
            st.session_state.loss_function,
        ),
        use_container_width=True,
        config={"displayModeBar": True},
    )


def _render_tabs(data, degree: int):
    tab_data, tab_history, tab_surface = st.tabs(
        ["Точки, на которых учимся", "Протокол по шагам", "Поверхность ошибки в 3D"]
    )

    with tab_data:
        st.caption(
            "Сами данные: x — признак, y — то, что модель должна предсказать. Эти точки "
            "порождены выбранной слева зависимостью с добавленным шумом."
        )
        st.dataframe(data, use_container_width=True)

    with tab_history:
        st.caption(
            "По одной строке на шаг: какими стали веса, какой получилась ошибка и насколько "
            "крут был склон. Полезно, когда хочется увидеть числа, а не график."
        )
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("Пока пусто — запустите обучение слева.")

    with tab_surface:
        active_params = active_param_names(degree)
        ensure_distinct_surface_params(active_params, st.session_state)
        col_x, col_y, col_resolution = st.columns([1, 1, 1])
        col_x.selectbox(
            "Вес по горизонтальной оси",
            active_params,
            index=active_params.index(st.session_state.surface_param_x),
            key="surface_param_x",
        )
        y_options = [name for name in active_params if name != st.session_state.surface_param_x] or active_params
        if st.session_state.surface_param_y not in y_options:
            st.session_state.surface_param_y = y_options[0]
        col_y.selectbox(
            "Вес по второй горизонтальной оси",
            y_options,
            index=y_options.index(st.session_state.surface_param_y),
            key="surface_param_y",
        )
        col_resolution.slider(
            "Насколько мелкая сетка",
            15,
            55,
            int(st.session_state.surface_resolution),
            5,
            key="surface_resolution",
        )
        st.caption(
            "Ошибка как ландшафт: по двум горизонтальным осям — значения двух весов, "
            "по высоте — ошибка при таких весах. Градиентный спуск и есть скатывание вниз "
            "по этому склону; пройденный путь нанесён на поверхность. У квадратичной и "
            "кубической модели весов больше двух, поэтому показан срез: невыбранные веса "
            "зафиксированы на текущих значениях."
        )
        st.plotly_chart(
            loss_surface_figure(
                data=data,
                history=st.session_state.history,
                current_params=params_vector(st.session_state),
                degree=degree,
                loss_function=st.session_state.loss_function,
                x_param=st.session_state.surface_param_x,
                y_param=st.session_state.surface_param_y,
                resolution=st.session_state.surface_resolution,
            ),
            use_container_width=True,
            config={"displayModeBar": True},
        )


def main():
    st.set_page_config(
        page_title="Тренажёр: градиентный спуск",
        page_icon="chart_with_downwards_trend",
        layout="wide",
    )
    init_state(st.session_state)

    st.title("Тренажёр: градиентный спуск")
    st.caption(
        "Как модель подбирает свои числа. Здесь это видно целиком: данные, кривая, "
        "шаги оптимизатора и падающая ошибка — на одном экране."
    )

    quick_start(
        st,
        purpose=(
            "Модель подбирает свои веса шаг за шагом — туда, где ошибка падает. Здесь это видно на кривой, "
            "которую подгоняют к точкам."
        ),
        steps=(
            "Нажмите слева **«Запустить»** и посмотрите, как падает график ошибки.",
            "Поставьте **«Размер шага»** 0.3 и запустите снова: спуск дойдёт до дна примерно за десять шагов.",
            "Поставьте 1: ошибка перестанет падать, потому что каждый шаг перепрыгивает через дно.",
        ),
        notice=(
            "размер шага решает всё. Слишком маленький почти не двигает веса, слишком большой не даёт "
            "спуститься."
        ),
    )
    how_to_read(
        st,
        what=(
            "Градиентным спуском обучается почти всё современное машинное обучение — от "
            "логистической регрессии до больших языковых моделей. Идея одна: у модели есть "
            "несколько чисел (веса), есть мера её ошибки, и веса много раз понемногу "
            "сдвигаются в ту сторону, где ошибка падает быстрее всего. Здесь модель самая "
            "простая — кривая, которую подгоняют к набору точек, — зато весь процесс можно "
            "потрогать руками."
        ),
        controls=(
            "Слева четыре группы. **Данные** — какую зависимость спрятать в точках, сколько "
            "их и насколько они разбросаны. **Модель** — какой кривой приближаем и с каких "
            "весов начинаем: ползунки w₀…w₃ двигаются вручную, так можно поставить спуск в "
            "заведомо плохую стартовую точку. **Оптимизатор** — размер шага и сколько точек "
            "смотреть перед одним шагом. **Длительность** — сколько шагов делать и когда "
            "считать, что процесс сошёлся. Дальше: «Запустить» — прогнать все шаги, "
            "«Один шаг» — сдвинуться ровно на один, «Показать анимацию» — проиграть спуск "
            "покадрово."
        ),
        watch=(
            "На графике ошибки кривая должна падать и выполаживаться — это и есть сходимость. "
            "Наверху два числа: ошибка и крутизна склона; у сошедшегося спуска крутизна близка "
            "к нулю. Стоит специально сломать обучение: поставьте размер шага 1 — каждый шаг "
            "перепрыгнет через дно, и ошибка перестанет падать; а если при этом выбрать одну "
            "случайную точку на шаг, она улетит в бесконечность. И наоборот, при шаге "
            "1e-5 спуск за тысячу шагов почти не сдвинется. Подбор размера шага — главная "
            "практическая трудность обучения, и тут она видна за полминуты."
        ),
        data=(
            "Точки генерируются на месте: выбранная зависимость плюс случайный шум. Поэтому "
            "правильный ответ известен заранее — он нарисован на графике, и с ним можно "
            "сравнивать то, что нашла модель. На реальных данных такой роскоши нет."
        ),
    )
    reference(
        st,
        "gradient_descent",
        "loss",
        "learning_rate",
        "batching",
        "gradient_norm",
        "convergence",
        "rmse_mae",
        "overfitting",
    )

    controls = _render_sidebar_controls()
    if controls["reset_clicked"]:
        reset_all_state(st.session_state)
        st.rerun()

    data = refresh_dataset(
        st.session_state,
        force_regenerate=controls["shuffle_now"],
        increment_seed=controls["shuffle_now"],
    )
    sync_training_config(st.session_state)
    degree = controls["degree"]

    if controls["run_clicked"]:
        apply_run(st.session_state, data, degree, controls["shuffle_each_epoch"])
    if controls["step_clicked"]:
        apply_step(st.session_state, data, degree, controls["shuffle_each_epoch"])
    if controls["animation_clicked"]:
        apply_animation_build(st.session_state, data, degree, controls["shuffle_each_epoch"])

    left, right = st.columns([2, 1])
    with right:
        _render_metrics_panel(data, degree)
    with left:
        _render_visual_panel(data, degree)

    _render_tabs(data, degree)


if __name__ == "__main__":
    main()
