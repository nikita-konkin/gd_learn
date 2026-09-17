# Gradient Descent Playground

Интерактивный учебный проект на `Streamlit` для изучения градиентного спуска на синтетических данных.

Проект умеет:

- генерировать данные из разных распределений;
- обучать линейную, квадратичную и кубическую модели;
- запускать `Batch GD`, `SGD` и `Mini-batch SGD`;
- оптимизировать по `MSE` или `MAE`;
- показывать график функции потерь;
- строить анимацию шагов градиентного спуска;
- показывать 3D-поверхность функции потерь и траекторию оптимизации;
- отслеживать расходимость и обрезать неустойчивую историю.

## Быстрый запуск

Установить зависимости:

```bash
pip install -r requirements-dev.txt
```

Запустить приложение из папки с программой:

```bash
streamlit run gradient_descent_playground_v3.py
```

Запустить тесты и линтер:

```bash
pytest
```

```bash
ruff check .
```

## Развёртывание на GitHub Pages

GitHub Pages отдаёт только статические файлы, а `Streamlit` обычно требует
Python-сервер. Поэтому сайт собирается через
[stlite](https://github.com/whitphx/stlite): он запускает CPython (Pyodide) и
`Streamlit` прямо в браузере через WebAssembly. Сервер не нужен, весь расчёт
идёт на стороне пользователя.

Собрать статический сайт локально:

```bash
python scripts/build_site.py --output dist
```

Посмотреть результат:

```bash
python -m http.server 8000 --directory dist
```

Что происходит при сборке:

- исходники (`gradient_descent_playground_v3.py` и пакет `gd_playground`)
  копируются в `dist/` без изменений — развёрнутое приложение всегда совпадает
  с репозиторием;
- из шаблона `web/index.template.html` генерируется `index.html` с манифестом
  `stlite` (список файлов, точка входа, зависимости);
- создаётся `.nojekyll`, чтобы GitHub Pages не пропускал файлы через Jekyll.

Версия `stlite` и зависимости для браузера закреплены в
`scripts/build_site.py`. `numpy` и `pandas` берутся из готовых сборок Pyodide,
`plotly` ставится через `micropip`.

### Первичная настройка репозитория

1. Репозиторий должен быть на **GitHub** — `GitHub Actions` и `GitHub Pages`
   не работают с других хостингов (например, с GitFlic).
2. В `Settings` → `Pages` → `Build and deployment` выбрать источник
   **GitHub Actions**.
3. Отправить коммит в `main` — пайплайн задеплоит сайт автоматически.

Адрес сайта: `https://<пользователь>.github.io/<репозиторий>/`.

## CI/CD

`.github/workflows/ci.yml` — запускается на каждый push в `main` и на каждый
pull request:

| Job | Что делает |
| --- | --- |
| `lint` | `ruff check` по всему репозиторию |
| `test` | `pytest` на Python 3.10, 3.11, 3.12 и 3.13 |
| `build` | собирает статический сайт и выгружает его как артефакт Pages |

`.github/workflows/deploy-pages.yml` — запускается на push в `main` и вручную
(`workflow_dispatch`). Он переиспользует пайплайн `ci.yml`, поэтому деплой
происходит только после успешных линтера, тестов и сборки.

## Структура проекта

```text
gd_learn
├── gradient_descent_playground_v3.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .github
│   └── workflows
│       ├── ci.yml
│       └── deploy-pages.yml
├── scripts
│   └── build_site.py
├── web
│   └── index.template.html
├── gd_playground
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── plotting.py
│   ├── state.py
│   ├── training.py
│   └── workflow.py
└── tests
    ├── conftest.py
    ├── test_app_smoke.py
    ├── test_data_and_state.py
    ├── test_model_and_training.py
    ├── test_plotting.py
    ├── test_site_build.py
    └── test_workflow_and_api.py
```

## Архитектура

`gradient_descent_playground_v3.py`

- Совместимый точечный вход.
- Реэкспортирует публичные функции из пакета `gd_playground`.
- Запускает `main()` при старте через `streamlit run`.

`gd_playground/app.py`

- Тонкий UI-слой на `Streamlit`.
- Собирает сайдбар, метрики, графики и вкладки.

`gd_playground/data.py`

- Генерация данных.
- Нормализация признака.
- Ограничения для защиты от расходимости.

`gd_playground/model.py`

- Полиномиальные признаки.
- Предсказание модели.
- Текстовое представление уравнения.

`gd_playground/training.py`

- Функции потерь.
- Градиенты.
- История оптимизации.
- Подготовка стабильной истории для визуализации.

`gd_playground/plotting.py`

- Все `Plotly`-графики.
- Анимация.
- 3D-поверхность потерь.

`gd_playground/state.py`

- Работа с `session_state`.
- Инициализация, сброс, обновление данных.

`gd_playground/workflow.py`

- Сценарии `Run`, `Step`, `Build animation`.
- Сводки истории и вычисление текущих метрик.

## Алгоритмические возможности

### Распределения данных

- `Linear`
- `Quadratic`
- `Cubic`
- `Sine`

Для каждого распределения генерируются:

- `x` — входные точки;
- `y_true` — истинная функция;
- `y` — наблюдения с шумом.

### Модели

- `Linear`
- `Quadratic`
- `Cubic`

Модель обучается на нормализованном признаке:

```text
z = (x - mean(x)) / std(x)
```

Это сделано для лучшей численной устойчивости, особенно для квадратичной и кубической регрессии.

### Оптимизаторы

- `Batch GD` — один шаг по всему датасету.
- `SGD` — один шаг по одной точке.
- `Mini-batch SGD` — один шаг по небольшой подвыборке.

### Функции потерь

- `MSE`
- `MAE`

Выбранная функция потерь используется:

- в вычислении градиента;
- в истории оптимизации;
- в критерии сходимости;
- в графике loss;
- в анимации;
- на 3D-поверхности потерь.

### Защита от расходимости

Если обучение становится численно нестабильным, проект:

- обнаруживает нечисловые значения;
- проверяет взрыв `MSE`, норм градиента и предсказаний;
- обрезает историю на последней стабильной итерации;
- показывает пользователю только корректные кадры и графики.

## Использование в интерфейсе

### Блок `Data`

- `Data point distribution` — выбирает форму истинной функции.
- `Number of data points` — число точек.
- `Noise` — уровень шума.
- `Random seed` — зерно генератора.
- `Shuffle / regenerate data` — принудительно обновляет набор данных.

### Блок `Model`

- `Model` — тип модели.
- `Bias / w0`, `w1`, `w2`, `w3` — коэффициенты модели.

### Блок `Optimizer`

- `Gradient descent type` — вид градиентного спуска.
- `Loss function for optimizer` — функция потерь для оптимизатора.
- `Learning rate` — шаг обучения.
- `Mini-batch size` — размер батча для мини-батч режима.
- `Shuffle points during SGD/mini-batch` — перемешивание выборки.
- `Iterations (Run)` — число итераций для обычного запуска.
- `Iterations (Animation)` — число итераций для анимации.
- `Convergence tolerance delta loss` — допуск для оценки сходимости.

### Кнопки

- `Run` — запустить обучение на заданное число итераций.
- `Step` — выполнить один шаг обучения.
- `Build animation` — построить историю для анимации.
- `Reset` — вернуть проект к значениям по умолчанию.

### Графики

- `Model Fit` — данные, истинная функция, текущая модель и остатки.
- `Loss vs Iterations` — выбранная loss по наблюдаемым данным и по `y_true`.
- `Gradient Descent Animation` — анимированная траектория обучения.
- `3D Loss Surface` — поверхность функции потерь по двум выбранным параметрам.

## Примеры использования из Python

### Импорт

```python
import numpy as np
import gradient_descent_playground_v3 as app
```

### Генерация данных

```python
data = app.generate_data(
    n_points=30,
    distribution="Quadratic",
    noise=0.8,
    seed=7,
)
```

### Один запуск оптимизации

```python
history = app.optimizer_history(
    data=data,
    params=np.array([10.0, 1.0, 0.0, 0.0]),
    degree=2,
    optimizer="Mini-batch SGD",
    learning_rate=0.01,
    iterations=100,
    batch_size=8,
    shuffle_each_epoch=True,
    seed=42,
    loss_function="MSE",
)
```

### Стабильная история для графиков

```python
stable_history, divergence_iteration = app.prepare_history_for_display(
    history=history,
    data=data,
    degree=2,
    loss_function="MSE",
)
```

### Текущие метрики модели

```python
metrics = app.full_dataset_metrics(
    data=data,
    params=np.array([10.0, 1.0, 0.0, 0.0]),
    degree=2,
    loss_function="MSE",
)
```

### Построение графиков

```python
fig_loss = app.loss_figure(
    history=stable_history,
    data=data,
    degree=2,
    tolerance=1e-6,
    loss_function="MSE",
)

fig_surface = app.loss_surface_figure(
    data=data,
    history=stable_history,
    current_params=np.array([10.0, 1.0, 0.0, 0.0]),
    degree=2,
    loss_function="MSE",
    x_param="w0",
    y_param="w1",
    resolution=25,
)
```

## Описание всех публичных функций

Ниже перечислены функции, которые доступны через `import gradient_descent_playground_v3 as app`.

### Конфигурация

`Defaults`

- Dataclass со значениями по умолчанию для UI и вычислений.

### Данные и численная устойчивость

`generate_data(n_points, distribution, noise, seed)`

- Генерирует `DataFrame` с колонками `x`, `y`, `y_true`.

`should_regenerate_data(data, n_points, distribution, noise, seed, previous_distribution=None, previous_noise=None, previous_seed=None)`

- Определяет, надо ли пересоздать датасет при изменении параметров.

`clamp_batch_size(batch_size, n_points)`

- Ограничивает размер мини-батча диапазоном `[2, n_points]`.

`feature_transform(x)`

- Возвращает `(center, scale)` для нормализации признака.

`feature_transform_from_data(data)`

- То же самое, но получает `x` из `DataFrame`.

`normalized_feature_text(transform)`

- Превращает `(center, scale)` в строку вида `z = (x - c) / s`.

`divergence_limits(data)`

- Возвращает численные лимиты для отслеживания расходимости.

### Модель и признаки

`degree_for_model(model_type)`

- Возвращает степень полинома для модели.

`active_param_names(degree)`

- Возвращает список активных параметров, например `["w0", "w1", "w2"]`.

`param_index(param_name)`

- Преобразует имя параметра `w2` в индекс `2`.

`design_matrix(x, degree)`

- Строит матрицу полиномиальных признаков.

`predict_values(x, params, degree, transform=None)`

- Считает предсказания модели.

`equation_text(params, degree, variable_name="x")`

- Возвращает строковое представление уравнения.

### Loss и оптимизация

`loss_function_options()`

- Возвращает список поддерживаемых функций потерь.

`optimization_loss_label(loss_function, prefix="Observed")`

- Возвращает подпись для графиков и метрик.

`loss_axis_title(loss_function)`

- Возвращает заголовок оси `Y` для графика loss.

`mse_loss(x, y, params, degree, transform=None)`

- Считает `MSE`.

`mae_loss(x, y, params, degree, transform=None)`

- Считает `MAE`.

`optimization_loss(x, y, params, degree, loss_function="MSE", transform=None)`

- Унифицированный вход для вычисления активной функции потерь.

`gradient(x, y, params, degree, loss_function="MSE", transform=None)`

- Возвращает градиент выбранной функции потерь.

`full_dataset_metrics(data, params, degree, loss_function="MSE", transform=None)`

- Возвращает словарь метрик по всему датасету:
- `optimization_loss`
- `true_optimization_loss`
- `mse`
- `rmse`
- `mae`
- `true_mse`
- `true_rmse`
- `grad_norm`
- `param_norm`

`optimizer_history(data, params, degree, optimizer, learning_rate, iterations, batch_size, shuffle_each_epoch, seed, loss_function="MSE")`

- Выполняет обучение и сохраняет историю всех итераций.

`params_from_row(row)`

- Восстанавливает вектор параметров из строки истории.

`recompute_history_metrics(history, data, degree, loss_function="MSE")`

- Пересчитывает метрики истории по полным данным, даже если старые значения были неверными.

`prepare_history_for_display(history, data, degree, loss_function="MSE")`

- Очищает историю от нестабильных точек и возвращает:
- стабильную историю;
- номер итерации расходимости или `None`.

`convergence_iteration(history, tolerance, metric_key="optimization_loss")`

- Оценивает первую итерацию, где изменение метрики стало меньше порога.

### Графики и визуализация

`animation_slider_label(iteration, max_iteration, max_labels=12)`

- Возвращает компактную подпись для шкалы анимации.

`metrics_text(row, degree, transform, loss_function="MSE")`

- Собирает текстовый блок метрик для текущего состояния.

`model_figure(data, params, degree, title_suffix="")`

- Строит график модели, истинной функции, наблюдений и остатков.

`parameter_axis_values(history, reference_params, param_name, resolution)`

- Подготавливает диапазон значений параметра для 3D-графика.

`loss_figure(history, data, degree, tolerance, divergence_iteration=None, loss_function="MSE")`

- Строит график loss по итерациям.

`animated_dashboard_figure(data, history, degree, tolerance, divergence_iteration=None, loss_function="MSE")`

- Строит анимированный дашборд обучения.

`loss_surface_figure(data, history, current_params, degree, loss_function, x_param, y_param, resolution=35)`

- Строит 3D-поверхность функции потерь и траекторию градиентного спуска.

### Работа с состоянием

`init_state(session_state=None)`

- Заполняет `session_state` значениями по умолчанию.

`clear_training_outputs(session_state=None)`

- Очищает историю, анимацию и флаг расходимости.

`reset_all_state(session_state=None)`

- Полный сброс состояния.

`params_vector(session_state=None)`

- Читает `w0..w3` из состояния и возвращает `numpy`-вектор.

`write_params(params, session_state=None)`

- Записывает параметры обратно в состояние.

`ensure_distinct_surface_params(param_names, session_state=None)`

- Гарантирует, что на 3D-графике выбраны два разных параметра.

`refresh_dataset(session_state=None, force_regenerate=False, increment_seed=False)`

- Обновляет датасет и при необходимости очищает историю.

`sync_training_config(session_state=None)`

- Отслеживает смену модели, оптимизатора или loss и сбрасывает устаревшую историю.

### Workflow-уровень

`run_history(data, params, degree, optimizer, learning_rate, iterations, batch_size, shuffle_each_epoch, seed, loss_function)`

- Высокоуровневый запуск истории с последующей стабилизацией.

`apply_run(session_state, data, degree, shuffle_each_epoch)`

- Реализует кнопку `Run`.

`apply_step(session_state, data, degree, shuffle_each_epoch)`

- Реализует кнопку `Step`.

`apply_animation_build(session_state, data, degree, shuffle_each_epoch)`

- Реализует кнопку `Build animation`.

`current_metrics(session_state, data, degree)`

- Возвращает текущие метрики и текущий трансформ признака.

`current_degree(session_state)`

- Возвращает степень текущей модели из состояния.

`history_summary(history, tolerance)`

- Возвращает краткую сводку по истории:
- лучшая loss;
- лучшая true-loss;
- лучшая `MSE`;
- номер сходимости;
- последняя loss.

### Точка входа

`main()`

- Главная функция Streamlit-приложения.

## Тесты

Тестовый набор покрывает:

- генерацию данных;
- нормализацию и устойчивость;
- вычисление loss и градиента;
- обучение для `Batch GD`, `SGD` и `Mini-batch SGD`;
- стабилизацию истории;
- построение графиков и анимации;
- 3D-график поверхности потерь;
- workflow-логику;
- публичный API;
- smoke-тест UI-слоя.

Запуск:

```powershell
pytest -q
```

## Практические сценарии для студентов

### 1. Сравнение оптимизаторов

- Выберите одинаковые данные и одинаковую модель.
- Запустите `Batch GD`, затем `SGD`, затем `Mini-batch SGD`.
- Сравните скорость снижения loss и форму траектории на 3D-графике.

### 2. Сравнение `MSE` и `MAE`

- Зафиксируйте данные и модель.
- Переключайте `Loss function for optimizer`.
- Смотрите, как меняются траектория, скорость обучения и итоговые параметры.

### 3. Численная устойчивость

- Увеличьте `Learning rate`.
- Перейдите на `Cubic` + `Mini-batch SGD`.
- Наблюдайте, когда обучение начинает расходиться и как приложение обрезает неустойчивую историю.

### 4. Анализ поверхности потерь

- Откройте вкладку `3D loss surface`.
- Выберите две координаты, например `w0` и `w1`.
- Смотрите, как траектория спуска зависит от типа loss и шага обучения.

## Ограничения

- 3D-график для квадратичной и кубической модели показывает только 2D-срез многомерной поверхности потерь.
- `MAE` имеет негладкую форму, поэтому траектория может выглядеть менее плавной, чем у `MSE`.
- Проект ориентирован на учебную визуализацию, а не на промышленное обучение моделей.
