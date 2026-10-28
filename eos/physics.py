"""Термодинамические соотношения и проверки (общие для всех моделей, numpy).

Используемые ограничения (ρ — плотность, T — температура, P — давление,
E — удельная внутренняя энергия):

1. Механическая устойчивость:  (∂P/∂ρ)_T > 0.
2. Термическая устойчивость:   C_v = (∂E/∂T)_ρ ≥ 0.
3. Скорость звука вещественна:  c_s² = (∂P/∂ρ)_T + T/(ρ² C_v)·(∂P/∂T)_ρ² > 0.
4. Причинность:                c_s < c.
5. Термодинамическая согласованность (соотношение Максвелла, следует из
   существования свободной энергии F(ρ,T)):
       ρ²(∂E/∂ρ)_T = P − T(∂P/∂T)_ρ .
   Безразмерная невязка  r_M = [ρ²∂E/∂ρ − P + T∂P/∂T] / P  (P > 0 в области).

Единицы: ГПа/(г/см³) = кДж/г = 10⁶ м²/с², поэтому c² = 8.98755e10 кДж/г.
"""
from __future__ import annotations

import numpy as np

C_LIGHT_SQ = 299_792_458.0 ** 2 / 1e6  # (м/с)² → кДж/г
CV_RTOL = 1e-6  # C_v < 0 считается нарушением, если T·C_v < −1e-6·max(|E|, 1 кДж/г)
                # (отсекает шум округления табличных данных на «плато» E(T))


def sound_speed_sq(rho, T, dPdr, dPdT, dEdT):
    """Квадрат адиабатической скорости звука с регуляризацией C_v ≥ 1e-30.

    Args:
        rho: плотность, г/см³.
        T: температура, K.
        dPdr, dPdT, dEdT: частные производные.

    Returns:
        np.ndarray: c_s², кДж/г (= км²/с²).
    """
    cv = np.maximum(dEdT, 1e-30)
    return dPdr + T / (rho ** 2 * cv) * dPdT ** 2


def sound_speed_sq_exact(rho, T, dPdr, dPdT, dEdT):
    """c_s² без регуляризации.

    При C_v = 0 и ∂P/∂T ≠ 0 тепловой член равен +∞, при C_v < 0 он отрицателен —
    так нарушение устойчивости проявляется в c_s².

    Args:
        rho: плотность, г/см³.
        T: температура, K.
        dPdr, dPdT, dEdT: частные производные.

    Returns:
        np.ndarray: c_s², кДж/г.
    """
    num = T * dPdT ** 2
    den = rho ** 2 * dEdT
    term = np.full(np.shape(num), np.inf)
    np.divide(num, den, out=term, where=den != 0)
    term = np.where(num == 0, 0.0, term)
    return dPdr + term


def maxwell_residual(rho, T, P, dPdT, dEdr):
    """Безразмерная невязка соотношения Максвелла.

    r_M = [ρ²∂E/∂ρ − P + T∂P/∂T] / |P|.

    Args:
        rho: плотность, г/см³.
        T: температура, K.
        P: давление, ГПа.
        dPdT: (∂P/∂T)_ρ, ГПа/K.
        dEdr: (∂E/∂ρ)_T, кДж·см³/г².

    Returns:
        np.ndarray: r_M (0 — полная согласованность).
    """
    return (rho ** 2 * dEdr - P + T * dPdT) / np.abs(P)


def violations(rho, T, P, E, dPdr, dPdT, dEdr, dEdT, maxwell_tol: float = 0.05) -> dict:
    """Флаги нарушений термодинамических ограничений в каждой точке.

    Args:
        rho, T: плотность (г/см³) и температура (K), форма (N,).
        P, E: давление (ГПа) и энергия (кДж/г), форма (N,).
        dPdr, dPdT, dEdr, dEdT: частные производные, форма (N,).
        maxwell_tol: допуск на |r_M|, выше которого точка считается
            термодинамически несогласованной.

    Returns:
        dict[str, np.ndarray[bool]]: ключи ``dPdrho_le0``, ``Cv_lt0``,
        ``cs2_le0``, ``cs_ge_c``, ``P_le0``, ``maxwell``, ``any_stability``.
    """
    cs2 = sound_speed_sq_exact(rho, T, dPdr, dPdT, dEdT)
    rM = maxwell_residual(rho, T, P, dPdT, dEdr)
    f = dict(
        P_le0=P <= 0,
        dPdrho_le0=dPdr <= 0,
        Cv_lt0=T * dEdT < -CV_RTOL * np.maximum(np.abs(E), 1.0),
        cs2_le0=cs2 <= 0,
        cs_ge_c=cs2 >= C_LIGHT_SQ,
        maxwell=np.abs(rM) > maxwell_tol,
    )
    f["any_stability"] = f["dPdrho_le0"] | f["Cv_lt0"] | f["cs2_le0"] | f["cs_ge_c"]
    return f


def summarize_violations(flags: dict, rM: np.ndarray | None = None) -> dict:
    """Доли нарушений и статистика невязки Максвелла.

    Args:
        flags: словарь булевых массивов из ``violations``.
        rM: невязка Максвелла (опционально).

    Returns:
        dict[str, float]: доли нарушений в %, ``maxwell_med`` и ``maxwell_p95``.
    """
    s = {k: 100.0 * float(np.mean(v)) for k, v in flags.items()}
    if rM is not None:
        a = np.abs(rM[np.isfinite(rM)])
        s.update(maxwell_med=float(np.median(a)), maxwell_p95=float(np.percentile(a, 95)))
    return s


def grid_derivatives(rho, T, P, E):
    """Производные табличных данных на сетке (центральные разности).

    Args:
        rho: (Nr,), T: (NT,), P, E: (Nr, NT).

    Returns:
        tuple: dPdr, dPdT, dEdr, dEdT, каждая формы (Nr, NT).
    """
    return (np.gradient(P, rho, axis=0), np.gradient(P, T, axis=1),
            np.gradient(E, rho, axis=0), np.gradient(E, T, axis=1))


def sample_domain(n: int, seed: int = 0, rho_range=(1e-3, 1e4), T_range=(0.1, 1e7)):
    """Случайные точки, равномерные по ln ρ и ln T.

    Args:
        n: число точек.
        seed: зерно генератора.
        rho_range: диапазон плотности, г/см³.
        T_range: диапазон температуры, K.

    Returns:
        tuple[np.ndarray, np.ndarray]: (rho, T), форма (n,).
    """
    g = np.random.default_rng(seed)
    rho = np.exp(g.uniform(*np.log(rho_range), n))
    T = np.exp(g.uniform(*np.log(T_range), n))
    return rho, T
