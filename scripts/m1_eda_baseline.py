"""Модуль 1: загрузка данных, EDA, сечения, проверка физики в самих таблицах,
билинейный бейзлайн (точность + скорость), коэффициенты нормирования.

Запуск:  python scripts/m1_eda_baseline.py --data data/raw --out artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos import physics as ph  # noqa: E402
from eos.baseline import BilinearEOS  # noqa: E402
from eos.data import Normalizer, load_all, make_dataset  # noqa: E402
from eos.metrics import accuracy, per_region  # noqa: E402
from eos.viz import DIV, INK2, SEQ, SERIES, plt, save  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

GT_COLOR = {"log-log": SERIES[0], "log-lin": SERIES[1], "lin-log": SERIES[2], "lin-lin": SERIES[6]}


def region_table(regions, out):
    """Таблица-паспорт 20 областей (тип сетки, размеры, диапазоны ρ, T, P, E).

    Args:
        regions: список областей.
        out: каталог артефактов.

    Returns:
        pandas.DataFrame: таблица (также сохраняется в tables/regions.csv).
    """
    rows = []
    for r in regions:
        rows.append(dict(region=r.name, grid=r.grid_type, n_rho=len(r.rho), n_T=len(r.T),
                         rho_min=r.rho[0], rho_max=r.rho[-1], T_min=r.T[0], T_max=r.T[-1],
                         P_min=r.P.min(), P_max=r.P.max(), E_min=r.E.min(), E_max=r.E.max()))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/tables/regions.csv", index=False)
    return df


def fig_region_map(regions, out):
    """Рисунок-карта 20 областей с типом равномерности сетки.

    Args:
        regions: список областей.
        out: каталог артефактов.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for r in regions:
        x0, x1 = np.log10(r.rho[[0, -1]]); y0, y1 = np.log10(r.T[[0, -1]])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=GT_COLOR[r.grid_type], alpha=.28,
                               ec="white", lw=2))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{r.name[3:]}\n{len(r.rho)}×{len(r.T)}",
                ha="center", va="center", fontsize=8, color="#0b0b0b")
    for g, c in GT_COLOR.items():
        ax.add_patch(Rectangle((0, 0), 0, 0, fc=c, alpha=.5, label=f"ρ-T: {g}"))
    ax.set(xlim=(-3, 4), ylim=(-1, 7), xlabel="lg ρ, г/см³", ylabel="lg T, K",
           title="20 табличных областей и тип равномерности сетки")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); ax.grid(False)
    save(fig, f"{out}/figures/m1_regions.png")


def fig_maps(regions, out):
    """Тепловые карты lg P и asinh E по всей области.

    Args:
        regions: список областей.
        out: каталог артефактов.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (ax, ttl) in enumerate(zip(axs, ["lg P, ГПа", "asinh(E / 1 кДж/г)"])):
        vals = [np.log10(r.P) if k == 0 else np.arcsinh(r.E) for r in regions]
        vmin = min(v.min() for v in vals); vmax = max(v.max() for v in vals)
        for r, v in zip(regions, vals):
            m = ax.pcolormesh(np.log10(r.rho), np.log10(r.T), v.T, cmap=SEQ, vmin=vmin, vmax=vmax,
                              shading="auto", rasterized=True)
        fig.colorbar(m, ax=ax, label=ttl)
        ax.set(xlabel="lg ρ, г/см³", ylabel="lg T, K", title=ttl); ax.grid(False)
    save(fig, f"{out}/figures/m1_maps_P_E.png")


def _line(regions, rho0=None, T0=None):
    """Сечение по узлам: при фиксированной ρ (ближайший узел) или T."""
    xs, P, E = [], [], []
    for r in regions:
        if rho0 is not None and r.rho[0] <= rho0 <= r.rho[-1]:
            i = np.argmin(np.abs(np.log(r.rho / rho0))); xs.append(r.T); P.append(r.P[i]); E.append(r.E[i])
        if T0 is not None and r.T[0] <= T0 <= r.T[-1]:
            j = np.argmin(np.abs(np.log(r.T / T0))); xs.append(r.rho); P.append(r.P[:, j]); E.append(r.E[:, j])
    x = np.concatenate(xs); o = np.argsort(x)
    return x[o], np.concatenate(P)[o], np.concatenate(E)[o]


def fig_sections(regions, out, model_fn=None, tag="m1"):
    """Сечения P и E при 4 фиксированных ρ и 4 фиксированных T.

    Args:
        regions: список областей.
        out: каталог артефактов.
        model_fn: функция (rho, T) → (P, E) для наложения модели (опционально).
        tag: префикс имени файла.
    """
    rhos = [1e-2, 0.5, 2.7, 100.0]; Ts = [1.0, 1e3, 3e4, 1e6]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for c, rho0 in enumerate(rhos):
        T, P, E = _line(regions, rho0=rho0)
        axs[0, 0].loglog(T, P, color=SERIES[c], label=f"ρ = {rho0:g}")
        axs[0, 1].semilogx(T, np.arcsinh(E), color=SERIES[c], label=f"ρ = {rho0:g}")
        if model_fn:
            Pm, Em = model_fn(np.full_like(T, rho0), T)
            axs[0, 0].loglog(T, Pm, "--", color=INK2, lw=1); axs[0, 1].semilogx(T, np.arcsinh(Em), "--", color=INK2, lw=1)
    for c, T0 in enumerate(Ts):
        rho, P, E = _line(regions, T0=T0)
        axs[1, 0].loglog(rho, P, color=SERIES[c], label=f"T = {T0:g} K")
        axs[1, 1].semilogx(rho, np.arcsinh(E), color=SERIES[c], label=f"T = {T0:g} K")
        if model_fn:
            Pm, Em = model_fn(rho, np.full_like(rho, T0))
            axs[1, 0].loglog(rho, Pm, "--", color=INK2, lw=1); axs[1, 1].semilogx(rho, np.arcsinh(Em), "--", color=INK2, lw=1)
    axs[0, 0].set(xlabel="T, K", ylabel="P, ГПа", title="P(T) при фиксированной ρ [г/см³]")
    axs[0, 1].set(xlabel="T, K", ylabel="asinh(E / 1 кДж/г)", title="E(T) при фиксированной ρ")
    axs[1, 0].set(xlabel="ρ, г/см³", ylabel="P, ГПа", title="P(ρ) при фиксированной T")
    axs[1, 1].set(xlabel="ρ, г/см³", ylabel="asinh(E / 1 кДж/г)", title="E(ρ) при фиксированной T")
    for ax in axs.flat:
        ax.legend(fontsize=8)
    if model_fn:
        fig.suptitle("Сплошные — таблица, пунктир — модель", color=INK2)
    save(fig, f"{out}/figures/{tag}_sections.png")


def data_physics(regions, out):
    """Проверка термодинамических ограничений на самих табличных данных.

    Args:
        regions: список областей.
        out: каталог артефактов.

    Returns:
        tuple[pandas.DataFrame, dict]: доли нарушений по областям и сводка невязки Максвелла.
    """
    res, rows = [], []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    norm = TwoSlopeNorm(0, -1, 1)
    for r in regions:
        dPdr, dPdT, dEdr, dEdT = ph.grid_derivatives(r.rho, r.T, r.P, r.E)
        R, TT = np.meshgrid(r.rho, r.T, indexing="ij")
        rM = ph.maxwell_residual(R, TT, r.P, dPdT, dEdr)
        inner = (slice(1, -1), slice(1, -1))
        f = ph.violations(R[inner], TT[inner], r.P[inner], r.E[inner], dPdr[inner], dPdT[inner],
                          dEdr[inner], dEdT[inner])
        s = ph.summarize_violations(f, rM[inner]); s["region"] = r.name; rows.append(s)
        axs[0].pcolormesh(np.log10(r.rho), np.log10(r.T), np.clip(rM, -1, 1).T, cmap=DIV, norm=norm,
                          shading="auto", rasterized=True)
        bad = np.zeros_like(r.P, bool); bad[inner] = f["Cv_lt0"] | f["dPdrho_le0"] | f["cs2_le0"]
        axs[1].pcolormesh(np.log10(r.rho), np.log10(r.T), np.where(bad, 1.0, np.nan).T, cmap=DIV,
                          vmin=-1, vmax=1, shading="auto", rasterized=True)
        res.append(rM[inner].ravel())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIV)
    fig.colorbar(sm, ax=axs[0], label="r_M = (ρ²∂E/∂ρ − P + T∂P/∂T) / P")
    axs[0].set(title="Невязка Максвелла в табличных данных", xlabel="lg ρ, г/см³", ylabel="lg T, K")
    axs[1].set(title="Узлы с нарушением устойчивости (C_v<0, ∂P/∂ρ≤0, c_s²≤0)",
               xlabel="lg ρ, г/см³", ylabel="lg T, K", xlim=(-3, 4), ylim=(-1, 7))
    for ax in axs:
        ax.grid(False)
    save(fig, f"{out}/figures/m1_data_physics.png")
    import pandas as pd
    df = pd.DataFrame(rows).set_index("region")
    df.round(4).to_csv(f"{out}/tables/data_physics_by_region.csv")
    allr = np.abs(np.concatenate(res))
    return df, dict(maxwell_med=float(np.median(allr)), maxwell_p95=float(np.percentile(allr, 95)),
                    maxwell_gt5pct=float(100 * np.mean(allr > 0.05)))


def fig_cold_and_step(regions, out):
    """Иллюстрации двух особенностей данных: холодная кривая и скачок при 143 K.

    Args:
        regions: список областей.
        out: каталог артефактов.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    # (а) холодная кривая: P и ρ²dE/dρ при минимальной T
    rho, P, E = _line(regions, T0=0.1)
    rho, idx = np.unique(rho, return_index=True); P, E = P[idx], E[idx]
    dEdr = np.gradient(E, rho)
    axs[0].loglog(rho, P, color=SERIES[0], label="P (таблица)")
    axs[0].loglog(rho, np.abs(rho ** 2 * dEdr), color=SERIES[1], label="|ρ² ∂E/∂ρ| (из таблицы E)")
    axs[0].set(xlabel="ρ, г/см³", ylabel="ГПа", title="T = 0.1 K: при согласованности кривые совпадали бы")
    axs[0].legend()
    # (б) ступенька при T ≈ 143 K
    r = [x for x in regions if x.name == "Obl12"][0]
    for c, i in enumerate([0, 100, 200, 299]):
        m = (r.T > 120) & (r.T < 170)
        axs[1].plot(r.T[m], r.E[i, m] - r.E[i, m][0], ".-", ms=4, color=SERIES[c], label=f"ρ = {r.rho[i]:.2f}")
    axs[1].set(xlabel="T, K", ylabel="E − E(120 K), кДж/г", title="Скачок E при T ≈ 143 K (Obl12): C_v < 0")
    axs[1].legend()
    save(fig, f"{out}/figures/m1_cold_curve_and_step.png")


def eval_model_physics(deriv: dict, rho, T) -> dict:
    """Сводка нарушений физики по словарю производных модели.

    Args:
        deriv: словарь P, E, dPdr, dPdT, dEdr, dEdT.
        rho: плотность, г/см³.
        T: температура, K.

    Returns:
        dict[str, float]: доли нарушений и статистика |r_M|.
    """
    f = ph.violations(rho, T, deriv["P"], deriv["E"], deriv["dPdr"], deriv["dPdT"], deriv["dEdr"], deriv["dEdT"])
    rM = ph.maxwell_residual(rho, T, deriv["P"], deriv["dPdT"], deriv["dEdr"])
    return ph.summarize_violations(f, rM)


def main():
    """Точка входа модуля 1: EDA, нормировка, бейзлайн, сводка в tables/m1_summary.json.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw"); ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    for d in ("figures", "tables", "models"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    regions = load_all(a.data)
    df = region_table(regions, a.out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    ds = make_dataset(regions)
    print({k: len(v["rho"]) for k, v in ds.items()})
    np.savez_compressed("data/processed/dataset.npz",
                        **{f"{k}_{c}": v for k, d in ds.items() for c, v in d.items()})
    norm = Normalizer.fit(**{c: ds["train"][c] for c in ("rho", "T", "P", "E")})
    norm.to_json(f"{a.out}/models/normalization.json")

    fig_region_map(regions, a.out); fig_maps(regions, a.out); fig_sections(regions, a.out)
    fig_cold_and_step(regions, a.out)
    phys_df, phys_all = data_physics(regions, a.out)
    print("Физика в данных:", phys_all)

    te = ds["test"]
    summary = {"data_physics": phys_all}
    rho_d, T_d = ph.sample_domain(100_000, seed=1)
    for values in ("raw", "log"):
        bl = BilinearEOS(regions, use="train", values=values)
        t0 = time.perf_counter(); P, E = bl.predict(te["rho"], te["T"]); _ = time.perf_counter() - t0
        acc = accuracy(te["P"], te["E"], P, E)
        # скорость: 10^5 случайных запросов
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); bl.predict(rho_d, T_d); ts.append(time.perf_counter() - t0)
        acc["time_1e5_ms"] = 1e3 * min(ts)
        acc["size_MB"] = sum(f.values.nbytes + sum(g.nbytes for g in f.grid)
                             for _, _, f in bl.items.values()) / 2 ** 20
        acc["phys_test"] = eval_model_physics(bl.derivatives(te["rho"], te["T"]), te["rho"], te["T"])
        acc["phys_domain"] = eval_model_physics(bl.derivatives(rho_d, T_d), rho_d, T_d)
        pr = per_region(te["region"], te["P"], te["E"], P, E)
        acc["per_region"] = pr
        summary[f"bilinear_{values}"] = acc
        print(f"bilinear[{values}]", {k: (round(v, 4) if isinstance(v, float) else '…') for k, v in acc.items()})
    bl = BilinearEOS(regions, use="all", values="raw")
    with open(f"{a.out}/tables/m1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
