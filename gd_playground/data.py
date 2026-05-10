from __future__ import annotations

import numpy as np
import pandas as pd


def generate_data(n_points: int, distribution: str, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-4.0, 4.0, int(n_points)))

    if distribution == "Linear":
        y_true = 2.2 * x + 12.0
    elif distribution == "Quadratic":
        y_true = 0.85 * x**2 - 1.6 * x + 8.0
    elif distribution == "Cubic":
        y_true = 0.18 * x**3 - 0.7 * x**2 + 1.5 * x + 10.0
    elif distribution == "Sine":
        y_true = 4.0 * np.sin(1.2 * x) + 10.0
    else:
        y_true = 2.2 * x + 12.0

    y = y_true + rng.normal(0.0, float(noise), int(n_points))
    return pd.DataFrame({"x": x, "y": y, "y_true": y_true})


def should_regenerate_data(
    data: pd.DataFrame | None,
    n_points: int,
    distribution: str,
    noise: float,
    seed: int,
    previous_distribution: str | None = None,
    previous_noise: float | None = None,
    previous_seed: int | None = None,
) -> bool:
    if data is None:
        return True

    return (
        len(data) != int(n_points)
        or previous_distribution != distribution
        or previous_noise != noise
        or previous_seed != seed
    )


def clamp_batch_size(batch_size: int, n_points: int) -> int:
    upper = max(2, int(n_points))
    return min(max(2, int(batch_size)), upper)


def feature_transform(x) -> tuple[float, float]:
    values = np.asarray(x, dtype=float)
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return center, scale


def feature_transform_from_data(data: pd.DataFrame) -> tuple[float, float]:
    return feature_transform(data["x"].to_numpy(dtype=float))


def normalized_feature_text(transform: tuple[float, float]) -> str:
    center, scale = transform
    return f"z = (x - {center:.3f}) / {scale:.3f}"


def divergence_limits(data: pd.DataFrame) -> dict[str, float]:
    y_values = np.concatenate(
        [data["y"].to_numpy(dtype=float), data["y_true"].to_numpy(dtype=float)]
    )
    y_scale = float(np.max(np.abs(y_values)))
    prediction_limit = max(1000.0, 50.0 * y_scale)
    return {
        "prediction_limit": prediction_limit,
        "mse_limit": prediction_limit**2,
        "grad_norm_limit": max(1e4, 500.0 * y_scale),
    }
