from __future__ import annotations

import pandas as pd
import streamlit as st

from gd_playground.config import (
    CONVERGENCE_OPTIONS,
    DATA_DISTRIBUTIONS,
    LEARNING_RATE_OPTIONS,
    MODEL_TYPES,
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
from gd_playground.training import loss_function_options, optimization_loss_label
from gd_playground.workflow import (
    apply_animation_build,
    apply_run,
    apply_step,
    current_metrics,
    history_summary,
)


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
            "Run",
            type="primary",
            use_container_width=True,
            help="Прогнать выбранное число итераций и показать итог.",
        )
        step_clicked = col_b.button(
            "Step",
            use_container_width=True,
            help="Один шаг градиентного спуска — видно, куда сдвинулись параметры.",
        )

        col_c, col_d = st.columns(2)
        animation_clicked = col_c.button(
            "Build animation",
            use_container_width=True,
            help="Собрать анимацию спуска с ползунком по итерациям.",
        )
        reset_clicked = col_d.button(
            "Reset",
            use_container_width=True,
            help="Вернуть все параметры и историю к значениям по умолчанию.",
        )

    return run_clicked, step_clicked, animation_clicked, reset_clicked


def _render_data_controls():
    with st.expander("Данные", expanded=True):
        st.session_state.distribution = st.selectbox(
            "Data point distribution",
            DATA_DISTRIBUTIONS,
            index=DATA_DISTRIBUTIONS.index(st.session_state.distribution),
            help="Форма зависимости, из которой генерируются точки.",
        )
        st.session_state.n_points = st.slider(
            "Number of data points",
            5,
            250,
            int(st.session_state.n_points),
            1,
        )
        st.session_state.noise = st.slider(
            "Noise",
            0.0,
            5.0,
            float(st.session_state.noise),
            0.1,
            help="Разброс точек вокруг истинной зависимости.",
        )
        st.session_state.seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.seed),
            step=1,
        )
        shuffle_now = st.button("Shuffle / regenerate data", use_container_width=True)
    return shuffle_now


def _render_model_controls():
    with st.expander("Модель", expanded=True):
        st.session_state.model_type = st.selectbox(
            "Model",
            MODEL_TYPES,
            index=MODEL_TYPES.index(st.session_state.model_type),
            help="Степень полинома, которым приближаем данные.",
        )
        degree = degree_for_model(st.session_state.model_type)
        st.session_state.w0 = st.slider("Bias / w0", -20.0, 30.0, float(st.session_state.w0), 0.01)
        st.session_state.w1 = st.slider("w1", -10.0, 10.0, float(st.session_state.w1), 0.01)

        if degree >= 2:
            st.session_state.w2 = st.slider("w2", -5.0, 5.0, float(st.session_state.w2), 0.01)
        else:
            st.session_state.w2 = 0.0

        if degree >= 3:
            st.session_state.w3 = st.slider("w3", -2.0, 2.0, float(st.session_state.w3), 0.01)
        else:
            st.session_state.w3 = 0.0
    return degree


def _render_optimizer_controls():
    with st.expander("Оптимизатор", expanded=True):
        st.session_state.optimizer = st.selectbox(
            "Gradient descent type",
            OPTIMIZERS,
            index=OPTIMIZERS.index(st.session_state.optimizer),
            help="Сколько точек участвует в одном обновлении параметров.",
        )
        losses = loss_function_options()
        st.session_state.loss_function = st.selectbox(
            "Loss function for optimizer",
            losses,
            index=losses.index(st.session_state.loss_function),
        )
        st.session_state.learning_rate = st.select_slider(
            "Learning rate",
            options=LEARNING_RATE_OPTIONS,
            value=float(st.session_state.learning_rate),
            format_func=lambda value: f"{value:.0e}" if value < 0.001 else f"{value:g}",
            help="Слишком большой шаг — расходимость, слишком маленький — медленно.",
        )

        if st.session_state.optimizer == "Mini-batch SGD":
            st.session_state.batch_size = st.slider(
                "Mini-batch size",
                2,
                max(2, int(st.session_state.n_points)),
                clamp_batch_size(st.session_state.batch_size, st.session_state.n_points),
                1,
            )
        elif st.session_state.optimizer == "SGD":
            st.session_state.batch_size = 1

        shuffle_each_epoch = st.checkbox("Shuffle points during SGD/mini-batch", value=True)
    return shuffle_each_epoch


def _render_run_length_controls():
    with st.expander("Длительность и сходимость", expanded=False):
        st.session_state.iterations_run = st.slider(
            "Iterations (Run)",
            1,
            3000,
            int(st.session_state.iterations_run),
            1,
        )
        st.session_state.iterations_animation = st.slider(
            "Iterations (Animation)",
            10,
            600,
            int(st.session_state.iterations_animation),
            10,
        )
        st.session_state.convergence_tolerance = st.select_slider(
            "Convergence tolerance delta loss",
            options=CONVERGENCE_OPTIONS,
            value=float(st.session_state.convergence_tolerance),
            format_func=lambda value: f"{value:.0e}",
            help="Насколько мало должно меняться значение потерь, чтобы считать это сходимостью.",
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
        st.caption(
            "Другие playground'ы: [метрики перевода](mt/) · "
            "[языковая модель](lm/) · [векторизация](vec/) · "
            "[память переводов](tm/)"
        )

    return {
        "degree": degree,
        "shuffle_now": shuffle_now,
        "shuffle_each_epoch": shuffle_each_epoch,
        "run_clicked": run_clicked,
        "step_clicked": step_clicked,
        "animation_clicked": animation_clicked,
        "reset_clicked": reset_clicked,
    }


def _render_training_status(summary):
    """The verdict on the run — above the table of numbers, not below it."""
    if summary is None:
        st.info("Нажмите **Run**, **Step** или **Build animation**, чтобы начать обучение.")
        return

    if st.session_state.divergence_iteration is not None:
        st.error(
            f"Обучение разошлось на итерации {st.session_state.divergence_iteration}. "
            f"Показана последняя устойчивая итерация: {summary['max_iteration']}. "
            "Уменьшите learning rate."
        )
    elif summary["convergence_iteration"] is None:
        st.warning(f"Сходимость не достигнута за {summary['max_iteration']} итераций.")
    else:
        st.success(f"Сошлось примерно за {summary['convergence_iteration']} итераций.")


def _render_metrics_panel(data, degree: int):
    metrics, transform = current_metrics(st.session_state, data, degree)
    summary = history_summary(st.session_state.history, st.session_state.convergence_tolerance)

    st.subheader("Что происходит")
    _render_training_status(summary)

    # The three numbers people watch first. The other seven used to sit here
    # in one solid column and drown; they live under an expander now.
    loss_label = optimization_loss_label(st.session_state.loss_function)
    top_left, top_right = st.columns(2)
    top_left.metric(loss_label, f"{metrics['optimization_loss']:.6f}")
    top_right.metric("Gradient norm", f"{metrics['grad_norm']:.6f}")
    st.caption(
        f"{st.session_state.model_type} · {st.session_state.optimizer} · "
        f"{st.session_state.loss_function}"
    )

    with st.expander("Все метрики", expanded=False):
        st.metric(
            optimization_loss_label(st.session_state.loss_function, "True-function"),
            f"{metrics['true_optimization_loss']:.6f}",
        )
        st.metric("Observed MSE", f"{metrics['mse']:.6f}")
        st.metric("True MSE", f"{metrics['true_mse']:.6f}")
        st.metric("RMSE", f"{metrics['rmse']:.6f}")
        st.metric("MAE", f"{metrics['mae']:.6f}")

    if summary is not None:
        with st.expander("Лучшее за историю обучения", expanded=False):
            loss_caption = (
                f"Last stable plotted {st.session_state.loss_function}"
                if st.session_state.divergence_iteration is not None
                else f"Loss graph final {st.session_state.loss_function}"
            )
            st.caption(f"{loss_caption}: {summary['last_optimization_loss']:.6f}")
            st.caption(
                f"Best {st.session_state.loss_function} in history: "
                f"{summary['best_optimization_loss']:.6f}"
            )
            st.caption(
                f"Best true-function {st.session_state.loss_function} in history: "
                f"{summary['best_true_optimization_loss']:.6f}"
            )
            st.caption(f"Best observed MSE in history: {summary['best_observed_mse']:.6f}")
            st.caption(f"Best true-function MSE in history: {summary['best_true_mse']:.6f}")

    params = params_vector(st.session_state)
    st.code(
        f"{equation_text(params, degree, 'z')}\n{normalized_feature_text(transform)}",
        language="text",
    )

    if st.session_state.optimizer == "Batch GD":
        meaning = "Использует все точки для каждого обновления параметров."
    elif st.session_state.optimizer == "SGD":
        meaning = "Использует одну точку для каждого обновления параметров."
    else:
        meaning = f"Использует {st.session_state.batch_size} точек для каждого обновления параметров."
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
            "Нажмите Play внутри графика. Анимация использует выбранную функцию "
            "потерь и держит метрики вне области построения."
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
    tab_data, tab_history, tab_surface = st.tabs(["Data", "Training history", "3D loss surface"])

    with tab_data:
        st.dataframe(data, use_container_width=True)

    with tab_history:
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("Нажмите Run, Step или Build animation.")

    with tab_surface:
        active_params = active_param_names(degree)
        ensure_distinct_surface_params(active_params, st.session_state)
        col_x, col_y, col_resolution = st.columns([1, 1, 1])
        col_x.selectbox(
            "X parameter",
            active_params,
            index=active_params.index(st.session_state.surface_param_x),
            key="surface_param_x",
        )
        y_options = [name for name in active_params if name != st.session_state.surface_param_x] or active_params
        if st.session_state.surface_param_y not in y_options:
            st.session_state.surface_param_y = y_options[0]
        col_y.selectbox(
            "Y parameter",
            y_options,
            index=y_options.index(st.session_state.surface_param_y),
            key="surface_param_y",
        )
        col_resolution.slider(
            "Surface resolution",
            15,
            55,
            int(st.session_state.surface_resolution),
            5,
            key="surface_resolution",
        )
        st.caption(
            "Для квадратичной и кубической модели это 2D-срез многомерной "
            "поверхности потерь: все невыбранные параметры зафиксированы на текущих значениях."
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
        page_title="Gradient Descent Playground",
        page_icon="chart_with_downwards_trend",
        layout="wide",
    )
    init_state(st.session_state)

    st.title("Interactive Exercise: Gradient Descent Playground")
    st.caption(
        "Слева задайте данные, модель и оптимизатор, затем нажмите **Run**. "
        "Batch GD, SGD и mini-batch SGD, нелинейные модели, анимация спуска "
        "и оценка числа итераций до сходимости."
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
