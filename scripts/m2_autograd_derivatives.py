"""Пункт 2.3 ТЗ: частные производные в тестовых точках через torch.autograd.

Для каждой модели считает ∂P/∂ρ, ∂P/∂T, ∂E/∂ρ, ∂E/∂T и c_s² через ``torch.autograd``
во всех test-точках, сохраняет выборку в CSV и сравнивает с явным якобианом (ONNX-путь).

Запуск: python scripts/m2_autograd_derivatives.py --models pinn pinn_thermo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos.model import EOSInference, autograd_derivatives  # noqa: E402
from eos.train import load_checkpoint, load_processed  # noqa: E402

A = "artifacts"
KEYS = ("dPdr", "dPdT", "dEdr", "dEdT", "cs2")


def main():
    """Точка входа: autograd-производные в test-точках, CSV-выборка и сравнение с явным якобианом."""
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["pinn"])
    ap.add_argument("--bs", type=int, default=50_000); ap.add_argument("--sample", type=int, default=5000)
    a = ap.parse_args()
    te = load_processed()["test"]
    report = {}
    for name in a.models:
        net, th, _, _ = load_checkpoint(f"{A}/models/{name}.pt")
        inf = EOSInference(net, th).eval()
        ag = {k: [] for k in ("P", "E") + KEYS}; ex = {k: [] for k in KEYS}
        for i in range(0, len(te["rho"]), a.bs):
            r = torch.tensor(te["rho"][i:i + a.bs], dtype=torch.float32); t = torch.tensor(te["T"][i:i + a.bs], dtype=torch.float32)
            d = autograd_derivatives(net, th, r, t)
            for k in ag:
                ag[k].append(d[k].double().numpy())
            with torch.no_grad():
                o = inf(r, t).double().numpy()
            for j, k in enumerate(KEYS):
                ex[k].append(o[:, 2 + j])
        ag = {k: np.concatenate(v) for k, v in ag.items()}; ex = {k: np.concatenate(v) for k, v in ex.items()}
        rel = {k: float(np.median(np.abs(ag[k] - ex[k]) / (np.abs(ag[k]) + 1e-30))) for k in KEYS}
        report[name] = dict(n_test_points=int(len(te["rho"])), median_rel_diff_autograd_vs_jacobian=rel)
        idx = np.random.default_rng(0).choice(len(te["rho"]), size=min(a.sample, len(te["rho"])), replace=False)
        df = pd.DataFrame({"rho": te["rho"][idx], "T": te["T"][idx], **{k: ag[k][idx] for k in ag}})
        df.to_csv(f"{A}/tables/autograd_derivatives_{name}.csv", index=False, float_format="%.6e")
        print(name, report[name])
    with open(f"{A}/tables/autograd_derivatives_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
