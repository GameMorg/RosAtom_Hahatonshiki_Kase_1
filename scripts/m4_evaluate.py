"""Модуль 4: оценка, проверка производных через autograd, экспорт (TorchScript/ONNX),
замеры размера и скорости, графики и итоговая таблица baseline / Vanilla / PINN.

Запуск:  python scripts/m4_evaluate.py --models vanilla pinn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos import physics as ph  # noqa: E402
from eos.baseline import BilinearEOS  # noqa: E402
from eos.data import load_all  # noqa: E402
from eos.metrics import accuracy, per_region  # noqa: E402
from eos.model import EOSInference, EOSPredict  # noqa: E402
from eos.train import load_checkpoint, load_processed  # noqa: E402
from eos.viz import DIV, INK2, SEQ, SERIES, plt, save  # noqa: E402

A = "artifacts"
N_SPEED = 100_000


def timeit(fn, reps=7):
    """Медианное время вызова функции (после прогрева).

    Args:
        fn: функция без аргументов.
        reps: число повторов.

    Returns:
        float: время, мс.
    """
    fn()  # прогрев
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def run_inf(mod, rho, T, bs=200_000):
    """Батчевый инференс экспортируемого модуля EOSInference.

    Args:
        mod: модуль (ρ, T) → (N, 7).
        rho: плотность, г/см³.
        T: температура, K.
        bs: размер батча.

    Returns:
        dict[str, np.ndarray]: P, E, dPdr, dPdT, dEdr, dEdT, cs2.
    """
    out = []
    with torch.inference_mode():
        for i in range(0, len(rho), bs):
            out.append(mod(torch.tensor(rho[i:i + bs], dtype=torch.float32),
                           torch.tensor(T[i:i + bs], dtype=torch.float32)).double().numpy())
    o = np.concatenate(out)
    return {k: o[:, i] for i, k in enumerate(("P", "E", "dPdr", "dPdT", "dEdr", "dEdT", "cs2"))}


def phys_summary(d, rho, T):
    """Нарушения физики по словарю производных.

    Args:
        d: словарь из ``run_inf``.
        rho: плотность, г/см³.
        T: температура, K.

    Returns:
        tuple[dict, dict]: сводка (в %) и флаги по точкам.
    """
    f = ph.violations(rho, T, d["P"], d["E"], d["dPdr"], d["dPdT"], d["dEdr"], d["dEdT"])
    return ph.summarize_violations(f, ph.maxwell_residual(rho, T, d["P"], d["dPdT"], d["dEdr"])), f


def autograd_check(net, th, rho, T):
    """Сверка явных производных с torch.autograd (п. 2.3 ТЗ).

    Args:
        net: сеть.
        th: модуль термодинамики.
        rho: плотность, г/см³.
        T: температура, K.

    Returns:
        tuple[dict, float]: медианные относительные расхождения по 4 производным и время autograd, мс.
    """
    r = torch.tensor(rho, dtype=torch.float32, requires_grad=True)
    t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
    P, E, _, _ = th.values(net(th.x(r, t)))
    dP = torch.autograd.grad(P.sum(), (r, t), retain_graph=True)
    dE = torch.autograd.grad(E.sum(), (r, t))
    ag = dict(dPdr=dP[0], dPdT=dP[1], dEdr=dE[0], dEdT=dE[1])
    ex = run_inf(EOSInference(net, th), rho, T)
    res = {}
    for k, v in ag.items():
        v = v.detach().double().numpy()
        res[k] = float(np.median(np.abs(v - ex[k]) / (np.abs(v) + 1e-30)))
    t_ag = timeit(lambda: autograd_derivs(net, th, *speed_points()))
    return res, t_ag


_SP = {}


def speed_points():
    """Фиксированный набор 10⁵ случайных точек для замеров скорости.

    Returns:
        tuple[np.ndarray, np.ndarray]: (rho, T).
    """
    if not _SP:
        _SP["p"] = ph.sample_domain(N_SPEED, seed=7)
    return _SP["p"]


def autograd_derivs(net, th, rho, T):
    """Производные через autograd (для замера времени).

    Args:
        net: сеть.
        th: модуль термодинамики.
        rho: плотность, г/см³.
        T: температура, K.
    """
    r = torch.tensor(rho, dtype=torch.float32, requires_grad=True)
    t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
    P, E, _, _ = th.values(net(th.x(r, t)))
    torch.autograd.grad(P.sum(), (r, t), retain_graph=True); torch.autograd.grad(E.sum(), (r, t))


def export(name, net, th):
    """Экспорт модели: TorchScript, ONNX (полный и только P, E), state_dict.

    Args:
        name: имя модели.
        net: сеть.
        th: модуль термодинамики.

    Returns:
        dict[str, str]: пути к ONNX-файлам (``full``, ``_predict``).
    """
    inf, prd = EOSInference(net, th).eval(), EOSPredict(net, th).eval()
    ex = (torch.rand(16) * 10 + 0.1, torch.rand(16) * 1e4 + 1)
    ts = torch.jit.trace(inf, ex); ts.save(f"{A}/models/{name}.ts.pt")
    paths = {}
    for mod, suffix in ((inf, ""), (prd, "_predict")):
        p = f"{A}/models/{name}{suffix}.onnx"
        kw = dict(input_names=["rho", "T"], output_names=["out"],
                  dynamic_axes={"rho": {0: "N"}, "T": {0: "N"}, "out": {0: "N"}}, opset_version=17)
        try:
            torch.onnx.export(mod, ex, p, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(mod, ex, p, **kw)
        paths[suffix or "full"] = p
    torch.save(net.state_dict(), f"{A}/models/{name}_state_dict.pt")
    return paths


def onnx_session(path):
    """Сессия ONNX Runtime с максимальной оптимизацией графа.

    Args:
        path: путь к .onnx.

    Returns:
        onnxruntime.InferenceSession: сессия на CPU.
    """
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def eval_nn(name, ds, rho_d, T_d):
    """Полная оценка нейросетевой модели: точность, физика, autograd, экспорт, размер, скорость.

    Args:
        name: имя модели.
        ds: train/val/test.
        rho_d: случайные плотности для проверки физики.
        T_d: случайные температуры.

    Returns:
        tuple: (результаты, предсказания на test, производные на случайных точках, флаги, (net, th)).
    """
    net, th, cfg, norm = load_checkpoint(f"{A}/models/{name}.pt")
    inf, prd = EOSInference(net, th).eval(), EOSPredict(net, th).eval()
    te = ds["test"]
    d = run_inf(inf, te["rho"], te["T"])
    res = accuracy(te["P"], te["E"], d["P"], d["E"])
    res["per_region"] = per_region(te["region"], te["P"], te["E"], d["P"], d["E"])
    res["phys_test"], _ = phys_summary(d, te["rho"], te["T"])
    dd = run_inf(inf, rho_d, T_d)
    res["phys_domain"], flags = phys_summary(dd, rho_d, T_d)
    ag, t_ag = autograd_check(net, th, te["rho"][:20000], te["T"][:20000])
    res["autograd_vs_explicit_rel"] = ag
    paths = export(name, net, th)
    res["size_MB_state_dict"] = os.path.getsize(f"{A}/models/{name}_state_dict.pt") / 2 ** 20
    res["size_MB_onnx"] = os.path.getsize(paths["full"]) / 2 ** 20
    res["size_MB_onnx_predict"] = os.path.getsize(paths["_predict"]) / 2 ** 20
    res["n_params"] = sum(p.numel() for p in net.parameters())
    r, t = speed_points()
    rt, tt = torch.tensor(r, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
    ts = torch.jit.load(f"{A}/models/{name}.ts.pt")
    s_full, s_pred = onnx_session(paths["full"]), onnx_session(paths["_predict"])
    feeds = {"rho": r.astype(np.float32), "T": t.astype(np.float32)}
    with torch.inference_mode():
        res["t_values_torch_ms"] = timeit(lambda: prd(rt, tt))
        res["t_derivs_torch_ms"] = timeit(lambda: inf(rt, tt))
        res["t_derivs_torchscript_ms"] = timeit(lambda: ts(rt, tt))
    res["t_derivs_autograd_ms"] = t_ag
    res["t_values_onnx_ms"] = timeit(lambda: s_pred.run(None, feeds))
    res["t_derivs_onnx_ms"] = timeit(lambda: s_full.run(None, feeds))
    # согласованность ONNX и torch
    o = s_full.run(None, feeds)[0]
    with torch.inference_mode():
        o2 = inf(rt, tt).numpy()
    res["onnx_vs_torch_max_rel_P"] = float(np.max(np.abs(o[:, 0] - o2[:, 0]) / o2[:, 0]))
    return res, d, dd, flags, (net, th)


def eval_baseline(regions, ds, rho_d, T_d, values):
    """Оценка билинейного бейзлайна.

    Args:
        regions: список областей.
        ds: train/val/test.
        rho_d: случайные плотности.
        T_d: случайные температуры.
        values: ``raw`` или ``log``.

    Returns:
        tuple: (результаты, флаги нарушений, интерполятор).
    """
    bl = BilinearEOS(regions, use="train", values=values)
    te = ds["test"]
    P, E = bl.predict(te["rho"], te["T"])
    res = accuracy(te["P"], te["E"], P, E)
    res["per_region"] = per_region(te["region"], te["P"], te["E"], P, E)
    res["phys_test"], _ = phys_summary(bl.derivatives(te["rho"], te["T"]), te["rho"], te["T"])
    dd = bl.derivatives(rho_d, T_d)
    res["phys_domain"], flags = phys_summary(dd, rho_d, T_d)
    r, t = speed_points()
    res["t_values_numpy_ms"] = timeit(lambda: bl.predict(r, t), reps=5)
    res["t_derivs_numpy_ms"] = timeit(lambda: bl.derivatives(r, t), reps=3)
    res["size_MB"] = sum(f.values.nbytes + sum(g.nbytes for g in f.grid) for _, _, f in bl.items.values()) / 2 ** 20
    res["size_MB_full_table"] = sum(BilinearEOS(regions, use="all").items[k][2].values.nbytes
                                    for k in range(1, 21)) / 2 ** 20
    return res, flags, bl


# ---------------- графики ----------------
def fig_learning(names):
    """Кривые обучения (данные, валидация, физические слагаемые).

    Args:
        names: имена моделей с файлами *_history.csv.
    """
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, n in enumerate(names):
        h = pd.read_csv(f"{A}/models/{n}_history.csv")
        axs[0].semilogy(h.epoch, h.data, color=SERIES[c], label=n)
        v = h.dropna(subset=["val_P_mape_pct"])
        axs[1].semilogy(v.epoch, v.val_P_mape_pct, "o-", ms=3, color=SERIES[c], label=f"{n}: P")
        axs[1].semilogy(v.epoch, v.val_E_mre_pct, "s--", ms=3, color=SERIES[c], label=f"{n}: E", alpha=.7)
        if h[["mech", "therm", "sound", "maxwell"]].sum().sum() > 0:
            for k, (col, ls) in enumerate(zip(["mech", "therm", "sound", "maxwell"], ["-", "--", ":", "-."])):
                axs[2].semilogy(h.epoch, h[col].clip(lower=1e-12), ls, color=SERIES[k + 2], label=f"{n}: {col}")
    axs[0].set(title="Ошибка на данных (MSE, норм.)", xlabel="эпоха")
    axs[1].set(title="Валидация: средняя отн. ошибка, %", xlabel="эпоха")
    axs[2].set(title="Физические слагаемые потерь", xlabel="эпоха")
    for ax in axs:
        ax.legend(fontsize=7.5)
    save(fig, f"{A}/figures/m2_learning_curves.png")


def fig_error_maps(ds, preds):
    """Тепловые карты относительных ошибок P и E на test.

    Args:
        ds: train/val/test.
        preds: {имя: {P, E}}.
    """
    te = ds["test"]
    fig, axs = plt.subplots(2, len(preds), figsize=(4.4 * len(preds), 7), squeeze=False)
    x, y = np.log10(te["rho"]), np.log10(te["T"])
    for c, (n, d) in enumerate(preds.items()):
        eP = np.log10(np.abs(d["P"] - te["P"]) / te["P"] + 1e-12)
        eE = np.log10(np.abs(d["E"] - te["E"]) / np.maximum(np.abs(te["E"]), 1) + 1e-12)
        for r, (e, lab) in enumerate(((eP, "P"), (eE, "E"))):
            hb = axs[r, c].hexbin(x, y, C=e, gridsize=70, cmap=SEQ, vmin=-6, vmax=-1, reduce_C_function=np.mean)
            axs[r, c].set(title=f"{n}: lg отн. ошибки {lab}", xlabel="lg ρ, г/см³", ylabel="lg T, K")
            axs[r, c].grid(False)
    fig.colorbar(hb, ax=axs, label="lg |Δ|/|·| (среднее в ячейке)", shrink=.8)
    save(fig, f"{A}/figures/m2_error_heatmaps.png")


def fig_violation_maps(rho_d, T_d, flags):
    """Карты нарушений устойчивости на случайных точках.

    Args:
        rho_d: плотности.
        T_d: температуры.
        flags: {имя: флаги нарушений}.
    """
    fig, axs = plt.subplots(1, len(flags), figsize=(4.4 * len(flags), 3.9), squeeze=False)
    x, y = np.log10(rho_d), np.log10(T_d)
    for c, (n, f) in enumerate(flags.items()):
        ax = axs[0, c]
        ax.scatter(x, y, s=.3, c="#e4e3df", rasterized=True)
        for k, (key, lab) in enumerate((("Cv_lt0", "C_v<0"), ("dPdrho_le0", "∂P/∂ρ≤0"),
                                        ("cs2_le0", "c_s²≤0"), ("cs_ge_c", "c_s≥c"))):
            m = f[key]
            ax.scatter(x[m], y[m], s=2, color=SERIES[[1, 0, 6, 3][k]], label=f"{lab}: {100 * m.mean():.2f}%",
                       rasterized=True)
        ax.set(title=f"{n}: нарушения устойчивости", xlabel="lg ρ, г/см³", ylabel="lg T, K", xlim=(-3, 4), ylim=(-1, 7))
        ax.legend(fontsize=7.5, loc="upper left", markerscale=4); ax.grid(False)
    save(fig, f"{A}/figures/m3_violation_maps.png")


def fig_maxwell_maps(rho_d, T_d, dds):
    """Карты невязки Максвелла.

    Args:
        rho_d: плотности.
        T_d: температуры.
        dds: {имя: производные}.
    """
    fig, axs = plt.subplots(1, len(dds), figsize=(4.4 * len(dds), 3.9), squeeze=False)
    from matplotlib.colors import TwoSlopeNorm
    nm = TwoSlopeNorm(0, -1, 1)
    for c, (n, d) in enumerate(dds.items()):
        rM = ph.maxwell_residual(rho_d, T_d, d["P"], d["dPdT"], d["dEdr"])
        hb = axs[0, c].hexbin(np.log10(rho_d), np.log10(T_d), C=np.clip(rM, -1, 1), gridsize=60, cmap=DIV, norm=nm)
        axs[0, c].set(title=f"{n}: невязка Максвелла r_M", xlabel="lg ρ, г/см³", ylabel="lg T, K"); axs[0, c].grid(False)
    fig.colorbar(hb, ax=axs, shrink=.85)
    save(fig, f"{A}/figures/m3_maxwell_maps.png")


def fig_pareto(rows):
    """Компромисс точность ↔ согласованность по Максвеллу.

    Args:
        rows: список словарей с полями lam, P_mape_pct, E_mre_pct, maxwell_med.
    """
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("lam")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df.maxwell_med, df.P_mape_pct, "o-", color=SERIES[0], label="P: MAPE")
    ax.plot(df.maxwell_med, df.E_mre_pct, "s-", color=SERIES[1], label="E: ср. отн. ошибка")
    for _, r in df.iterrows():
        ax.annotate(f"λ={r.lam:g}", (r.maxwell_med, r.P_mape_pct), fontsize=7.5, color=INK2,
                    xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="медиана |r_M| (несогласованность по Максвеллу)",
           ylabel="ошибка на test, %", title="Компромисс точность ↔ согласованность (свип λ_Maxwell)")
    ax.legend()
    save(fig, f"{A}/figures/m3_pareto_maxwell.png")
    df.to_csv(f"{A}/tables/pareto_maxwell.csv", index=False)


def main():
    """Точка входа модуля 4: оценка всех моделей, экспорт, графики, comparison.md.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "pinn"])
    ap.add_argument("--data", default="data/raw")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    regions = load_all(a.data)
    ds = load_processed()
    rho_d, T_d = ph.sample_domain(200_000, seed=1)
    results, preds, dds, flags = {}, {}, {}, {}
    for v in ("raw", "log"):
        results[f"bilinear_{v}"], flags[f"bilinear_{v}"], bl = eval_baseline(regions, ds, rho_d, T_d, v)
    dds["bilinear_log"] = bl.derivatives(rho_d, T_d)
    model_fns = {}
    for n in a.models:
        results[n], preds[n], dds[n], flags[n], (net, th) = eval_nn(n, ds, rho_d, T_d)
        model_fns[n] = (net, th)
        print(n, {k: v for k, v in results[n].items() if not isinstance(v, dict)})
    with open(f"{A}/tables/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)

    # итоговая таблица
    rows = []
    for n, r in results.items():
        pd_ = r["phys_domain"]
        rows.append({
            "model": n,
            "P MAPE, %": r["P_mape_pct"], "P p95, %": r["P_p95_pct"], "P max, %": r["P_max_pct"],
            "E MRE, %": r["E_mre_pct"], "E p95, %": r["E_p95_pct"],
            "∂P/∂ρ≤0, %": pd_["dPdrho_le0"], "C_v<0, %": pd_["Cv_lt0"], "c_s²≤0, %": pd_["cs2_le0"],
            "c_s≥c, %": pd_["cs_ge_c"], "любое наруш. устойч., %": pd_["any_stability"],
            "|r_M| медиана": pd_["maxwell_med"],
            "размер, МБ": r.get("size_MB", r.get("size_MB_state_dict")),
            "ONNX P,E / +произв., МБ": (f'{r["size_MB_onnx_predict"]:.2f} / {r["size_MB_onnx"]:.2f}' if "size_MB_onnx" in r else "—"),
            "10⁵ запросов P,E, мс": r.get("t_values_numpy_ms", r.get("t_values_onnx_ms")),
            "10⁵ запросов +производные, мс": r.get("t_derivs_numpy_ms", r.get("t_derivs_onnx_ms")),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{A}/tables/comparison.csv", index=False)
    with open(f"{A}/tables/comparison.md", "w", encoding="utf-8") as f:
        f.write(tab.to_markdown(index=False, floatfmt=".4g"))
    print(tab.to_string(index=False))

    # графики
    fig_learning([n for n in a.models if os.path.exists(f"{A}/models/{n}_history.csv")])
    fig_error_maps(ds, {"bilinear_log": dict(zip(("P", "E"), BilinearEOS(regions, values="log").predict(
        ds["test"]["rho"], ds["test"]["T"]))), **preds})
    fig_violation_maps(rho_d, T_d, {k: flags[k] for k in ["bilinear_log", *a.models]})
    fig_maxwell_maps(rho_d, T_d, {k: dds[k] for k in ["bilinear_log", *a.models]})
    from m1_eda_baseline import fig_sections  # noqa: E402
    for n, (net, th) in model_fns.items():
        prd = EOSPredict(net, th).eval()
        def fn(r, t, prd=prd):
            with torch.inference_mode():
                o = prd(torch.tensor(r, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)).double().numpy()
            return o[:, 0], o[:, 1]
        fig_sections(regions, A, model_fn=fn, tag=f"m3_{n}")

    # Парето по λ_Maxwell
    prow = []
    for p in sorted(glob.glob(f"{A}/models/sweep/*.pt")):
        net, th, cfg, _ = load_checkpoint(p)
        inf = EOSInference(net, th).eval()
        te = ds["test"]; d = run_inf(inf, te["rho"], te["T"])
        acc = accuracy(te["P"], te["E"], d["P"], d["E"])
        s, _ = phys_summary(run_inf(inf, rho_d, T_d), rho_d, T_d)
        prow.append(dict(lam=cfg.lam_maxwell, P_mape_pct=acc["P_mape_pct"], E_mre_pct=acc["E_mre_pct"],
                         maxwell_med=s["maxwell_med"], maxwell_p95=s["maxwell_p95"], stab=s["any_stability"]))
    fig_pareto(prow)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
