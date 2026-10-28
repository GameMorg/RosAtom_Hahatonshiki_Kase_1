"""Модули 2–3: обучение Vanilla-сети и PINN, свип по весу условия Максвелла.

Запуск:
    python scripts/m2_m3_train.py --config configs/vanilla.json
    python scripts/m2_m3_train.py --config configs/pinn.json
    python scripts/m2_m3_train.py --config configs/pinn.json --sweep-maxwell 0 1e-5 1e-4 1e-3 1e-2 --epochs 120
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos.data import Normalizer  # noqa: E402
from eos.train import config_from_json, load_processed, train  # noqa: E402


def main():
    """Точка входа модулей 2–3: обучение по JSON-конфигу или серия по λ_Maxwell.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="artifacts/models")
    ap.add_argument("--epochs", type=int, default=None, help="переопределить число эпох")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--sweep-maxwell", type=float, nargs="*", default=None,
                    help="обучить серию PINN с разными λ_Maxwell (кривая Парето)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = config_from_json(a.config)
    if a.epochs:
        cfg = replace(cfg, epochs=a.epochs)
    cfg = replace(cfg, threads=a.threads)
    norm = Normalizer.from_json("artifacts/models/normalization.json")
    ds = load_processed()
    if a.sweep_maxwell is None:
        train(cfg, ds, norm, out_prefix=f"{a.out}/{cfg.name}")
    else:
        os.makedirs(f"{a.out}/sweep", exist_ok=True)
        for lam in a.sweep_maxwell:
            c = replace(cfg, name=f"pinn_lamM_{lam:g}", lam_maxwell=lam)
            train(c, ds, norm, out_prefix=f"{a.out}/sweep/{c.name}")


if __name__ == "__main__":
    main()
