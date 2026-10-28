"""Дополнительные термодинамические проверки (диагностика, без обучения).

1. Фундаментальная производная (выпуклость изоэнтроп, условие Бете–Вейля):
       𝒢 = 1 + (ρ / c_s)(∂c_s/∂ρ)_S = 1 + ρ/(2c_s²) · (∂c_s²/∂ρ)_S ,
       (∂/∂ρ)_S = (∂/∂ρ)_T + (dT/dρ)_S (∂/∂T)_ρ ,   (dT/dρ)_S = T (∂P/∂T)_ρ / (ρ² C_v).
   При 𝒢 > 0 ударные волны сжатия устойчивы, волны разрежения — непрерывны.
2. Третье начало (Нернст): при T → 0 должно быть C_v → 0 и (∂P/∂T)_ρ → 0;
   мера — g_T = ∂lnP/∂lnT при T < 1 K.

Для сетей вторые производные считаются autograd'ом поверх явного якобиана,
для таблиц — центральными разностями по узлам.

Запуск: python scripts/m3_extra_physics.py --models vanilla pinn pinn_consistent
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos import physics as ph  # noqa: E402
from eos.data import load_all  # noqa: E402
from eos.model import EOSInference  # noqa: E402
from eos.train import load_checkpoint  # noqa: E402

A = "artifacts"


def fundamental_nn(name, rho, T, bs=20000):
    """Фундаментальная производная 𝒢 и g_T = ∂lnP/∂lnT для нейросетевой модели.

    Args:
        name: имя модели в artifacts/models (без .pt).
        rho: плотность, г/см³, форма (N,).
        T: температура, K, форма (N,).
        bs: размер батча.

    Returns:
        tuple[np.ndarray, np.ndarray]: (𝒢, g_T), форма (N,).
    """
    net, th, _, _ = load_checkpoint(f"{A}/models/{name}.pt")
    net = net.double(); th = th.double()
    inf = EOSInference(net, th)
    G, gT = [], []
    for i in range(0, len(rho), bs):
        r = torch.tensor(rho[i:i + bs], dtype=torch.float64, requires_grad=True)
        t = torch.tensor(T[i:i + bs], dtype=torch.float64, requires_grad=True)
        o = inf(r, t)
        P, dPdT, dEdT, cs2 = o[:, 0], o[:, 3], o[:, 5], o[:, 6]
        dc_r, dc_T = torch.autograd.grad(cs2.sum(), (r, t))
        dTdr_S = t * dPdT / (r ** 2 * dEdT)
        g = 1 + r / (2 * cs2) * (dc_r + dTdr_S * dc_T)
        G.append(g.detach().numpy()); gT.append((t * dPdT / P).detach().numpy())
    return np.concatenate(G), np.concatenate(gT)


def fundamental_data(regions):
    """𝒢 и g_T для табличных данных (центральные разности по узлам).

    Args:
        regions: список областей.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (𝒢, g_T, T) по внутренним узлам.
    """
    G, gT, Tall = [], [], []
    for reg in regions:
        dPdr, dPdT, dEdr, dEdT = ph.grid_derivatives(reg.rho, reg.T, reg.P, reg.E)
        R, TT = np.meshgrid(reg.rho, reg.T, indexing="ij")
        with np.errstate(all="ignore"):
            cs2 = dPdr + TT * dPdT ** 2 / (R ** 2 * dEdT)
            dc_r = np.gradient(cs2, reg.rho, axis=0); dc_T = np.gradient(cs2, reg.T, axis=1)
            g = 1 + R / (2 * cs2) * (dc_r + TT * dPdT / (R ** 2 * dEdT) * dc_T)
        inner = (slice(2, -2), slice(2, -2))
        G.append(g[inner].ravel()); gT.append((TT * dPdT / reg.P)[inner].ravel()); Tall.append(TT[inner].ravel())
    return np.concatenate(G), np.concatenate(gT), np.concatenate(Tall)


def summarize(G, gT, T):
    """Сводка: доля 𝒢 < 0 и |g_T| при T < 1 K.

    Args:
        G: фундаментальная производная.
        gT: ∂lnP/∂lnT.
        T: температура, K.

    Returns:
        dict[str, float]: G_neg_pct, G_nan_pct, G_median, lowT_gT_median, lowT_gT_p95.
    """
    ok = np.isfinite(G)
    low = T < 1.0
    return dict(G_neg_pct=float(100 * np.mean(G[ok] < 0)), G_nan_pct=float(100 * np.mean(~ok)),
                G_median=float(np.median(G[ok])),
                lowT_gT_median=float(np.median(np.abs(gT[low]))) if low.any() else float("nan"),
                lowT_gT_p95=float(np.percentile(np.abs(gT[low]), 95)) if low.any() else float("nan"))


def main():
    """Точка входа: дополнительные проверки для таблиц и моделей, tables/extra_physics.json.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "pinn"])
    ap.add_argument("--n", type=int, default=100_000)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    res = {}
    G, gT, T = fundamental_data(load_all("data/raw"))
    res["table (FD)"] = summarize(G, gT, T)
    rho, T = ph.sample_domain(a.n, seed=11)
    for n in a.models:
        G, gT = fundamental_nn(n, rho, T)
        res[n] = summarize(G, gT, T)
    for k, v in res.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    with open(f"{A}/tables/extra_physics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
