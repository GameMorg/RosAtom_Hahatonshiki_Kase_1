"""Замер скорости на 10⁸ случайных запросов (батчами по 10⁶) — ONNX Runtime и билинейная таблица.

Запуск: python scripts/m4_bench_1e8.py --models pinn pinn_thermo pinn_small
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
from eos.data import load_all  # noqa: E402

A = "artifacts"


def bench(fn, n_total=100_000_000, batch=1_000_000):
    """Время обработки n_total запросов батчами (одна и та же пачка случайных точек).

    Args:
        fn: функция (rho, T) → что угодно.
        n_total: общее число запросов.
        batch: размер батча.

    Returns:
        float: суммарное время, с.
    """
    rho, T = ph.sample_domain(batch, seed=5)
    fn(rho[:1000], T[:1000])
    t0 = time.perf_counter()
    for _ in range(n_total // batch):
        fn(rho, T)
    return time.perf_counter() - t0


def main():
    """Точка входа: время 10⁸ запросов для ONNX-моделей и бейзлайна, tables/bench_1e8.json."""
    import onnxruntime as ort
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["pinn"])
    ap.add_argument("--n", type=int, default=100_000_000)
    a = ap.parse_args()
    res = {}
    for n in a.models:
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(f"{A}/models/{n}_predict.onnx", so, providers=["CPUExecutionProvider"])
        res[f"{n} (ONNX, P,E)"] = bench(lambda r, t: sess.run(None, {"rho": r.astype(np.float32), "T": t.astype(np.float32)}), a.n)
        print(n, res[f"{n} (ONNX, P,E)"], flush=True)
    bl = BilinearEOS(load_all("data/raw"), values="log")
    res["bilinear (scipy, P,E)"] = bench(bl.predict, a.n)
    print(res)
    with open(f"{A}/tables/bench_1e8.json", "w", encoding="utf-8") as f:
        json.dump({k: {"seconds": v, "per_query_ns": v / a.n * 1e9} for k, v in res.items()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
