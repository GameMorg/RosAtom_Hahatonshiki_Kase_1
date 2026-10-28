"""Полный пайплайн: EDA и бейзлайн → Vanilla → PINN (+свип λ_Maxwell) → оценка/экспорт → сжатие.

Примеры:
    python run.py                       # всё с нуля (на CPU ≈ 1.5–2 ч)
    python run.py --skip-train          # только оценка уже обученных моделей
    python run.py --fast                # быстрый прогон для проверки (мало эпох)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def sh(*args):
    """Запускает Python-скрипт пайплайна и прерывается при ошибке.

    Args:
        *args: путь к скрипту и его аргументы.

    Raises:
        subprocess.CalledProcessError: если скрипт завершился с ошибкой.
    """
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    """Точка входа: разбирает аргументы и последовательно запускает модули 1–5.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw", help="каталог с Obl01.txt … Obl20.txt")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-sweep", action="store_true", help="не обучать серию PINN по λ_Maxwell")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    ep = ["--epochs", "5"] if a.fast else []
    sh("scripts/m1_eda_baseline.py", "--data", a.data)
    if not a.skip_train:
        sh("scripts/m2_m3_train.py", "--config", "configs/vanilla.json", *ep)
        # серия PINN по λ_Maxwell (дообучение Vanilla); вариант 1e-4 — старт для финального PINN
        lams = ["1e-4"] if a.no_sweep else ["0", "1e-4", "1e-3", "1e-2", "1e-1"]
        sh("scripts/m2_m3_train.py", "--config", "configs/pinn_finetune_sweep.json", "--sweep-maxwell", *lams,
           *(["--epochs", "2"] if a.fast else []))
        sh("scripts/m2_m3_train.py", "--config", "configs/pinn.json", *ep)          # PINN-A (из λ=1e-4)
        if not a.no_sweep:
            sh("scripts/m2_m3_train.py", "--config", "configs/pinn_thermo.json", *ep)  # PINN-T (из λ=1e-1)
        sh("scripts/m2_m3_train.py", "--config", "configs/pinn_small.json", *ep)
    models = ["vanilla", "pinn"] + (["pinn_thermo"] if os.path.exists("artifacts/models/pinn_thermo.pt") else [])
    sh("scripts/m4_evaluate.py", "--models", *models, "--data", a.data)
    sh("scripts/m2_autograd_derivatives.py", "--models", *models[1:])  # п. 2.3: autograd в test-точках
    sh("scripts/m3_extra_physics.py", "--models", *models)
    sh("scripts/m5_compress.py", "--model", "pinn")
    sh("examples/api_example.py")


if __name__ == "__main__":
    main()
