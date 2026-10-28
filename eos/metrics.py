"""Метрики точности для P и E."""
from __future__ import annotations

import numpy as np

E_FLOOR = 1.0  # кДж/г: масштаб для относительной ошибки E (E меняет знак)


def accuracy(P_true, E_true, P_pred, E_pred) -> dict:
    """Метрики точности.

    * для P (P > 0, меняется на 12 порядков) — относительная ошибка
      |ΔP|/P: среднее (MAPE), медиана, 95-й перцентиль, максимум, а также RMSE ln P;
    * для E (меняет знак) — относительная ошибка с «полом»
      |ΔE| / max(|E|, 1 кДж/г) и абсолютная MAE.

    Returns:
        dict[str, float]: ошибки в процентах (``*_pct``) и в абсолютных единицах.
    """
    rP = np.abs(P_pred - P_true) / np.abs(P_true)
    rE = np.abs(E_pred - E_true) / np.maximum(np.abs(E_true), E_FLOOR)
    return dict(
        P_mape_pct=100 * rP.mean(), P_med_pct=100 * np.median(rP),
        P_p95_pct=100 * np.percentile(rP, 95), P_max_pct=100 * rP.max(),
        P_rmse_ln=float(np.sqrt(np.mean((np.log(P_pred) - np.log(P_true)) ** 2))),
        E_mre_pct=100 * rE.mean(), E_med_pct=100 * np.median(rE),
        E_p95_pct=100 * np.percentile(rE, 95), E_max_pct=100 * rE.max(),
        E_mae=float(np.mean(np.abs(E_pred - E_true))),
    )


def per_region(region, P_true, E_true, P_pred, E_pred) -> dict[int, dict]:
    """Метрики точности отдельно по каждой из 20 областей.

    Args:
        region: номер области для каждой точки, форма (N,).
        P_true, E_true: табличные значения, форма (N,).
        P_pred, E_pred: предсказания модели, форма (N,).

    Returns:
        dict[int, dict]: {номер области: результат ``accuracy``}.
    """
    return {int(k): accuracy(P_true[m], E_true[m], P_pred[m], E_pred[m])
            for k in np.unique(region) for m in [region == k]}
