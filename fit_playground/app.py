"""Streamlit interface for the model-complexity playground.

Three tabs, three handles on the same question — how flexible should a model
be — each opening where the trouble is plainest. The tabs share no
control: the degree moves only the polynomial, the neighbours and the depth only
their own panel, the penalty only the regression.
"""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

from playground_common.compat import patch_pyarrow_stub

# Before the first sklearn import: stlite's pyarrow stub is missing types that
# sklearn's input check reaches for on every call.
patch_pyarrow_stub()

from fit_playground.models import (  # noqa: E402  — must follow the stub patch
    ALPHAS,
    DEFAULT_DEPTH,
    MAX_DEGREE,
    ZERO,
    Classifier,
    Penalty,
    PolynomialFit,
    correlation,
    feature_names,
    fit_polynomial,
    fresh_data,
    mean_prediction_error,
    moons,
    neighbours,
    penalty,
    polynomial_data,
    tree,
)
from fit_playground.plotting import (  # noqa: E402
    boundary_figure,
    degree_curve_figure,
    paths_figure,
    polynomial_figure,
)
from playground_common.explain import how_to_read, quick_start, reference  # noqa: E402
from playground_common.links import other_playgrounds  # noqa: E402
from playground_common.wording import as_printed, plural  # noqa: E402

TITLE = "Сложность модели"

DEGREE_LABEL = "Степень многочлена"
NEIGHBOURS_LABEL = "Число соседей, k"
DEPTH_LABEL = "Глубина дерева"
ALPHA_LABEL = "Сила штрафа за большие веса, alpha"

UNLIMITED = "без ограничения"
DEPTHS = ["1", "2", "3", "4", "5", "6", "8", "10", UNLIMITED]
DEFAULT_DEGREE = 17
DEFAULT_NEIGHBOURS = 1
DEFAULT_ALPHA = ALPHAS[-1]

# The ten columns of the diabetes table, as scikit-learn's description of the set names them;
# the abbreviation in brackets is the one in the original study.
FEATURE_MEANINGS = {
    "age": "возраст, лет",
    "sex": "пол",
    "bmi": "индекс массы тела: вес, делённый на квадрат роста",
    "bp": "среднее артериальное давление",
    "s1": "общий холестерин в сыворотке крови (tc)",
    "s2": "липопротеины низкой плотности, «плохой» холестерин (ldl)",
    "s3": "липопротеины высокой плотности, «хороший» холестерин (hdl)",
    "s4": "отношение общего холестерина к «хорошему» (tch)",
    "s5": "вероятно, логарифм уровня триглицеридов (ltg)",
    "s6": "уровень сахара в крови (glu)",
}


@lru_cache(maxsize=MAX_DEGREE + 1)
def _polynomial(degree: int) -> PolynomialFit:
    return fit_polynomial(degree)


@lru_cache(maxsize=128)
def _neighbours(count: int) -> Classifier:
    return neighbours(count)


@lru_cache(maxsize=16)
def _tree(depth: int | None) -> Classifier:
    return tree(depth)


@lru_cache(maxsize=len(ALPHAS))
def _penalty(alpha: float) -> Penalty:
    return penalty(alpha)


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.3g}"


def _controls() -> tuple[int, int, int | None, float]:
    """The sidebar: one control per tab, and each reaches only its own tab."""
    st.sidebar.header("Многочлен")
    degree = st.sidebar.slider(
        DEGREE_LABEL, 1, MAX_DEGREE, DEFAULT_DEGREE,
        help=(
            "Насколько гибкой разрешено быть кривой. Степень 1 — прямая, степень 17 — кривая, способная пройти почти "
            "через любые точки."
        ),
    )

    st.sidebar.header("Соседи и дерево")
    count = st.sidebar.slider(
        NEIGHBOURS_LABEL, 1, 99, DEFAULT_NEIGHBOURS,
        help="Чтобы классифицировать новую точку, модель находит k ближайших известных точек и "
        "выбирает класс большинства. При k = 1 ответ зависит от одной точки — это крайняя гибкость; при k = 99 "
        "модель усредняет треть всей выборки.",
    )
    depth_text = st.sidebar.select_slider(
        DEPTH_LABEL, DEPTHS, value=str(DEFAULT_DEPTH),
        help="Сколько последовательных проверок дереву разрешено сделать. Без ограничения оно дробит "
        "данные, пока в каждом листе не останется один класс, т. е. запоминает выборку.",
    )

    st.sidebar.header("Штраф")
    labels = [_alpha_label(alpha) for alpha in ALPHAS]
    chosen = st.sidebar.select_slider(
        ALPHA_LABEL, labels, value=_alpha_label(DEFAULT_ALPHA),
        help=(
            "Штраф удерживает модель от слишком больших весов. При нуле его нет, при большом штрафе модель "
            "вырождается в одно число для всех."
        ),
    )

    st.sidebar.divider()
    st.sidebar.markdown(other_playgrounds("ml-practice/fit"))

    depth = None if depth_text == UNLIMITED else int(depth_text)
    return degree, count, depth, ALPHAS[labels.index(chosen)]


def _error(value: float) -> str:
    """Three decimals for errors on the scale of the data, one for errors far beyond it.

    A degree-17 fit is badly conditioned: the browser's NumPy and a desktop one
    agree on its training error to the digit but part ways in the fourth figure
    of its error on new points (13.156 against 13.155). One decimal is all such
    a number can honestly carry.
    """
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def _polynomial_tab(degree: int) -> None:
    st.markdown("Двадцать пять точек синусоиды с шумом и многочлены разной степени.")
    fit = _polynomial(degree)
    baseline = mean_prediction_error()

    left, middle, right = st.columns(3)
    left.metric(
        "Ошибка на обучении",
        f"{fit.train_error:.3f}",
        help="Расхождение с теми двадцатью пятью точками, по которым модель подбиралась. "
        "Это число всегда можно уменьшить, взяв более гибкую кривую, поэтому само по себе оно "
        "ничего не доказывает.",
    )
    middle.metric(
        "Ошибка на новых точках",
        _error(fit.fresh_error),
        help="Расхождение с точками из того же закона, которых модель не видела. "
        "Только по этому числу видно, усвоила модель закономерность или шум.",
    )
    right.metric(
        "Ответ «среднее» на новых точках",
        f"{baseline:.3f}",
        help="Для сравнения: ошибка простейшей модели, которая на любой вход отвечает одним "
        "средним значением. Полезная модель должна быть по крайней мере не хуже.",
    )

    train_x, train_y = polynomial_data()
    fresh_x, fresh_y = fresh_data()
    st.plotly_chart(polynomial_figure(fit, train_x, train_y, fresh_x, fresh_y), use_container_width=True)

    if fit.fresh_error > baseline:
        times = fit.fresh_error / baseline
        st.warning(
            f"Ошибка на обучении {fit.train_error:.3f} — а на новых точках {_error(fit.fresh_error)}, "
            f"в {times:.0f} {plural(round(times), 'раз', 'раза', 'раз')} хуже, чем если бы модель всегда отвечала "
            "средним. Модель усвоила шум, а не закон."
        )

    degrees = list(range(1, MAX_DEGREE + 1))
    st.plotly_chart(
        degree_curve_figure(
            degrees,
            [_polynomial(value).train_error for value in degrees],
            [_polynomial(value).fresh_error for value in degrees],
            baseline,
            degree,
        ),
        use_container_width=True,
    )
    st.caption(
        "Ошибка на обучении убывает с каждой степенью, но ничего не доказывает. Ошибка на новых точках "
        "сначала убывает вместе с ней, затем начинает расти: многочлен повторяет шум."
    )


def _classifier_column(model: Classifier, title: str) -> None:
    features, target = moons()
    st.plotly_chart(boundary_figure(model, features, target, title), use_container_width=True)
    left, right = st.columns(2)
    left.metric(
        f"{title}: на обучении",
        as_printed(model.train_accuracy),
        help="Доля правильных ответов на тех точках, по которым строилась модель.",
    )
    right.metric(
        f"{title}: на проверке",
        as_printed(model.cv_accuracy),
        help="Доля правильных ответов на точках, которых модель не видела (среднее по пяти "
        "блокам кросс-валидации). Разрыв между двумя числами и есть переобучение.",
    )


def _neighbours_tab(count: int, depth: int | None) -> None:
    st.markdown(
        "Метод ближайших соседей и дерево решений на трёхстах точках двух полумесяцев. "
        "«На проверке» — доля правильных ответов, усреднённая по пяти блокам кросс-валидации."
    )
    knn = _neighbours(count)
    model = _tree(depth)
    left, right = st.columns(2)
    with left:
        _classifier_column(knn, f"k = {count}")
    with right:
        _classifier_column(model, f"глубина {depth if depth is not None else '∞'}")

    best = _neighbours(15)
    if count == 1:
        st.caption(
            f"При одном соседе модель отвечает на обучении безошибочно ({as_printed(knn.train_accuracy)}): "
            "каждая точка — сама себе сосед. Это переобучение, и граница действительно "
            f"огибает каждый выброс. Но на проверке k = 1 даёт {as_printed(knn.cv_accuracy)} против "
            f"{as_printed(best.cv_accuracy)} у k = 15 — цена переобучения здесь мала. "
            f"Сдвиньте k к 99: недообучение обходится в {as_printed(best.cv_accuracy - _neighbours(99).cv_accuracy)}."
        )
    st.caption(
        f"Дерево глубины {depth if depth is not None else 'без ограничения'}: "
        f"на обучении {as_printed(model.train_accuracy)}, на проверке {as_printed(model.cv_accuracy)}. "
        "Разница между этими числами — то, что по обучающей выборке увидеть нельзя."
    )


def _weight(value: float) -> float:
    """One decimal, and a weight Lasso has zeroed printed as 0 rather than −0."""
    return 0.0 if abs(value) < ZERO else round(value, 1)


def _features_expander(chosen: Penalty, names: tuple[str, ...]) -> None:
    """What each line on the chart measures, and where it stands at the chosen penalty."""
    with st.expander("Что означают признаки"):
        st.dataframe(
            {
                "признак": list(names),
                "что измерено": [FEATURE_MEANINGS[name] for name in names],
                "вес, Ridge": [_weight(value) for value in chosen.ridge_weights],
                "вес, Lasso": [_weight(value) for value in chosen.lasso_weights],
            },
            use_container_width=True,
            hide_index=True,
        )
        # The largest penalty each feature survives in Lasso.
        survives = {
            name: max((value for value in ALPHAS if abs(_penalty(value).lasso_weights[index]) >= ZERO), default=0.0)
            for index, name in enumerate(names)
        }
        first, second = sorted(names, key=survives.get, reverse=True)[:2]
        weakest = _penalty(ALPHAS[0])
        s1, s2 = (weakest.ridge_weights[names.index(name)] for name in ("s1", "s2"))
        st.caption(
            f"Веса указаны при выбранной alpha = {_alpha_label(chosen.alpha)}. "
            "В наборе 442 пациента с диабетом. Модель предсказывает, насколько болезнь продвинулась за год "
            "после измерений: это число от 25 до 346, и чем оно больше, тем хуже. Признаки перед обучением "
            "стандартизованы, поэтому вес показывает, на сколько изменится прогноз, если признак вырастет на "
            "одно стандартное отклонение, а веса разных признаков сопоставимы между собой.\n\n"
            f"Дольше всех Lasso сохраняет **{first}** и **{second}**: при сильном штрафе в модели остаются только "
            f"они. Признаки **s1** и **s2** почти дублируют друг друга (корреляция "
            f"{correlation('s1', 's2'):.2f}), поэтому при самом слабом штрафе Ridge назначает им большие веса "
            f"разного знака, {s1:.0f} и {s2:+.0f}, которые во многом компенсируют друг друга. Такие пары штраф "
            "сжимает в первую очередь."
        )


def _penalty_tab(alpha: float) -> None:
    st.markdown(
        "Ridge и Lasso на наборе данных diabetes. По десяти признакам пациента — возрасту, полу, индексу массы "
        "тела, давлению и шести анализам крови — модель предсказывает, как продвинется диабет за год. "
        "Признаки стандартизованы, метрика — R² на отложенных 30 % строк. Расшифровка признаков — под графиком."
    )
    chosen = _penalty(alpha)
    ridge_paths = [_penalty(value).ridge_weights for value in ALPHAS]
    lasso_paths = [_penalty(value).lasso_weights for value in ALPHAS]
    names = feature_names()

    left, middle, right = st.columns(3)
    left.metric(
        "Lasso обнулил признаков",
        f"{chosen.lasso_zeroed} из {len(names)}",
        help="Нулевой вес означает, что признак исключён из модели. Ridge этого не делает: "
        "он только сжимает веса, не обнуляя их.",
    )
    middle.metric(
        "R² Ridge",
        f"{chosen.ridge_r2:.3f}",
        help="Доля разброса целевой величины, которую модель объяснила, на отложенных 30 %. "
        "1 — идеальное предсказание, 0 — не лучше ответа средним значением.",
    )
    right.metric("R² Lasso", f"{chosen.lasso_r2:.3f}")

    st.plotly_chart(paths_figure(ALPHAS, ridge_paths, lasso_paths, names, alpha), use_container_width=True)
    _features_expander(chosen, names)

    if 0 < chosen.lasso_zeroed < len(names):
        kept = len(names) - chosen.lasso_zeroed
        st.info(
            f"Lasso обнулил {chosen.lasso_zeroed} {plural(chosen.lasso_zeroed, 'признак', 'признака', 'признаков')} "
            f"из {len(names)} и оставил {kept}: это и есть отбор признаков. Ridge не обнуляет ни одного — "
            "сравните их R²."
        )
    if chosen.lasso_zeroed == len(names):
        first = next(value for value in ALPHAS if _penalty(value).lasso_zeroed == len(names))
        st.warning(
            f"Обнулены все {len(names)} признаков, и R² Lasso — {chosen.lasso_r2:.3f}: модель отвечает одним "
            "числом для всех, и это уже не отбор признаков, а отказ от модели. На этой сетке так происходит "
            f"начиная с alpha ≈ {_alpha_label(first)}. Отбор, ради которого Lasso применяют, виден левее: "
            "сдвиньте штраф к 10."
        )
    st.caption(
        "Одна и та же alpha задаёт для Ridge и Lasso разную силу штрафа: Lasso в scikit-learn делит ошибку на "
        "удвоенный размер выборки, Ridge — нет. Поэтому на общей шкале Lasso успевает обнулить всё, пока Ridge "
        "лишь вдвое сжимает веса. Сравнение по одной alpha сопоставляет шкалы, а не методы."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "Слишком простая модель недообучается, слишком гибкая запоминает шум. Здесь гибкость задаётся "
        "настройкой, а цена ошибки измеряется на данных, которых модель не видела."
    )

    quick_start(
        st,
        purpose=(
            "Показать, какую гибкость стоит разрешить модели: слишком жёсткая не улавливает закономерность, "
            "слишком гибкая запоминает шум."
        ),
        steps=(
            "На вкладке «Многочлен» уменьшите **«Степень многочлена»** с 17 до 3: ошибка на новых точках "
            "снизится с 13.2 до 0.077.",
            "При каждой степени сравнивайте ошибку на обучении и на новых точках.",
            "Повторите на вкладках «Соседи и дерево» и «Штраф».",
        ),
        notice="с ростом гибкости ошибка на обучении только убывает, поэтому судить о модели можно лишь по новым точкам.",
    )
    how_to_read(
        st,
        what=(
            "Главный выбор в машинном обучении — какую гибкость разрешить модели. "
            "Слишком жёсткая не улавливает закономерность — это недообучение. Слишком гибкая "
            "проходит ровно через обучающие точки, включая случайный шум в них, и на новых "
            "данных работает плохо — это переобучение. Три вкладки показывают одно и то же "
            "явление на разных моделях: на кривой, подгоняемой к точкам; на методе ближайших "
            "соседей и дереве решений; на линейной регрессии со штрафом за большие веса."
        ),
        controls=(
            "У каждой вкладки свои настройки, и на другие вкладки они не влияют. **Степень "
            "многочлена** задаёт гибкость кривой. **Число соседей k** и **глубина дерева** — гибкость "
            "двух классификаторов: малое k и большая глубина означают большую гибкость. "
            "**Сила штрафа alpha** действует наоборот: чем больше штраф, тем жёстче модель."
        ),
        watch=(
            "На каждой вкладке сравнивайте два числа: на обучении и на новых данных (на второй "
            "вкладке — «на проверке»). Первое с ростом гибкости только улучшается, поэтому по нему "
            "нельзя судить о модели. Второе сначала улучшается, затем ухудшается; на нижнем графике "
            "первой вкладки этот перелом виден целиком, и его низшая точка соответствует гибкости, "
            "которую стоило выбрать. При "
            "степени 17 обратите внимание на предупреждение: модель оказывается хуже, чем "
            "если бы она на любой вход отвечала одним средним значением."
        ),
        data=(
            "Первая вкладка — двадцать пять точек синусоиды с шумом; проверочные точки взяты "
            "из того же закона отдельным генератором. Вторая — триста точек «двух полумесяцев», "
            "стандартный учебный набор. Третья — реальный набор данных diabetes из scikit-learn: "
            "442 пациента, десять признаков и показатель развития болезни через год; расшифровка "
            "признаков приведена на той же вкладке под графиком."
        ),
    )
    reference(
        st,
        "overfitting",
        "holdout",
        "cross_validation",
        "accuracy",
        "r2",
        "knn",
        "decision_tree",
        "regularisation",
        "ridge_lasso",
    )

    degree, count, depth, alpha = _controls()
    polynomial, classifiers, penalties = st.tabs(["Многочлен", "Соседи и дерево", "Штраф"])
    with polynomial:
        _polynomial_tab(degree)
    with classifiers:
        _neighbours_tab(count, depth)
    with penalties:
        _penalty_tab(alpha)

    with st.expander("Чего эта площадка не делает"):
        st.markdown(
            "- Не подбирает гиперпараметры автоматически: подбор по сетке с вложенной проверкой — "
            "отдельная тема.\n"
            "- Не меняет данные: точки одни и те же при любых настройках. Новые точки "
            "многочлена взяты из того же закона отдельным генератором.\n"
            "- Не сравнивает ансамбли: случайный лес и бустинг обучаются в браузере "
            "заметно дольше, а нового о сложности модели не показывают.\n"
            f"- Не нормирует штраф: сетка alpha общая для обеих моделей — от {_alpha_label(ALPHAS[0])} "
            f"до {_alpha_label(ALPHAS[-1])}, а её неодинаковый смысл для Ridge и Lasso показан без поправок."
        )


if __name__ == "__main__":
    main()
