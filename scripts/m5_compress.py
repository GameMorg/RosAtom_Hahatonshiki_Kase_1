"""Модуль 5: сжатие и ускорение инференса.

Сравниваемые варианты (для модели --model, по умолчанию pinn):
  * производные через autograd (эталон) vs явный forward-mode якобиан;
  * eager PyTorch vs TorchScript (freeze + optimize_for_inference) vs ONNX Runtime;
  * хранение весов в fp16 (размер ×0.5) — точность после обратного приведения к fp32;
  * динамическая int8-квантизация линейных слоёв (только значения P, E);
  * число потоков.

Запуск:  python scripts/m5_compress.py --model pinn
"""
from __future__ import annotations

import argparse
import copy
import io
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos import physics as ph  # noqa: E402
from eos.metrics import accuracy  # noqa: E402
from eos.model import EOSInference, EOSPredict  # noqa: E402
from eos.train import load_checkpoint, load_processed  # noqa: E402
from m4_evaluate import A, autograd_derivs, export, onnx_session, phys_summary, run_inf, timeit  # noqa: E402


def state_size_mb(sd) -> float:
    """Размер сериализованного state_dict.

    Args:
        sd: state_dict.

    Returns:
        float: размер, МБ.
    """
    buf = io.BytesIO(); torch.save(sd, buf)
    return buf.getbuffer().nbytes / 2 ** 20


def main():
    """Точка входа модуля 5: варианты сжатия и ускорения, compression.md.
    """
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pinn")
    a = ap.parse_args()
    net, th, cfg, norm = load_checkpoint(f"{A}/models/{a.model}.pt")
    ds = load_processed(); te = ds["test"]
    rho_d, T_d = ph.sample_domain(100_000, seed=7)
    rt, tt = torch.tensor(rho_d, dtype=torch.float32), torch.tensor(T_d, dtype=torch.float32)
    rows = []

    def add(variant, size, t_val, t_der, mod_der=None, mod_val=None):
        r = dict(variant=variant, size_MB=size, t_values_ms=t_val, t_derivs_ms=t_der)
        if mod_der is not None:
            d = run_inf(mod_der, te["rho"], te["T"])
            acc = accuracy(te["P"], te["E"], d["P"], d["E"])
            s, _ = phys_summary(run_inf(mod_der, rho_d, T_d), rho_d, T_d)
            r.update(P_mape_pct=acc["P_mape_pct"], E_mre_pct=acc["E_mre_pct"], stab_viol_pct=s["any_stability"])
        elif mod_val is not None:
            with torch.inference_mode():
                o = mod_val(torch.tensor(te["rho"], dtype=torch.float32), torch.tensor(te["T"], dtype=torch.float32)).double().numpy()
            acc = accuracy(te["P"], te["E"], o[:, 0], o[:, 1])
            r.update(P_mape_pct=acc["P_mape_pct"], E_mre_pct=acc["E_mre_pct"])
        rows.append(r); print(r, flush=True)

    size32 = state_size_mb(net.state_dict())
    inf, prd = EOSInference(net, th).eval(), EOSPredict(net, th).eval()
    with torch.inference_mode():
        t_val = timeit(lambda: prd(rt, tt)); t_der = timeit(lambda: inf(rt, tt))
    t_ag = timeit(lambda: autograd_derivs(net, th, rho_d, T_d))
    add("eager fp32, производные autograd", size32, t_val, t_ag)
    add("eager fp32, явный якобиан", size32, t_val, t_der, mod_der=inf)

    # TorchScript + freeze
    ex = (rt[:16], tt[:16])
    ts_inf = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(inf, ex).eval()))
    ts_prd = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(prd, ex).eval()))
    with torch.inference_mode():
        add("TorchScript (freeze+optimize)", size32, timeit(lambda: ts_prd(rt, tt)), timeit(lambda: ts_inf(rt, tt)))

    # ONNX Runtime
    full, light = onnx_session(f"{A}/models/{a.model}.onnx"), onnx_session(f"{A}/models/{a.model}_predict.onnx")
    feeds = {"rho": rho_d.astype(np.float32), "T": T_d.astype(np.float32)}
    add("ONNX Runtime", os.path.getsize(f"{A}/models/{a.model}.onnx") / 2 ** 20,
        timeit(lambda: light.run(None, feeds)), timeit(lambda: full.run(None, feeds)))

    # fp16-хранение весов
    sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
    torch.save(sd16, f"{A}/models/{a.model}_fp16_state_dict.pt")
    net16 = copy.deepcopy(net); net16.load_state_dict({k: v.float() for k, v in sd16.items()})
    add("веса fp16 (хранение), расчёт fp32", state_size_mb(sd16), t_val, t_der,
        mod_der=EOSInference(net16, th).eval())

    # динамическая int8-квантизация (значения)
    try:
        from torch.ao.quantization import quantize_dynamic
        qnet = quantize_dynamic(copy.deepcopy(net), {torch.nn.Linear}, dtype=torch.qint8)
        qprd = EOSPredict(qnet, th).eval()
        with torch.inference_mode():
            tq = timeit(lambda: qprd(rt, tt))
        add("int8 dynamic quant (только P,E)", state_size_mb(qnet.state_dict()), tq, float("nan"), mod_val=qprd)
    except Exception as e:  # noqa: BLE001
        print("int8 quantization недоступна:", e)

    # компактная PINN (64×4, 16 Фурье-признаков), обученная с теми же физическими потерями
    if os.path.exists(f"{A}/models/pinn_small.pt"):
        snet, sth, _, _ = load_checkpoint(f"{A}/models/pinn_small.pt")
        sinf, sprd = EOSInference(snet, sth).eval(), EOSPredict(snet, sth).eval()
        paths = export("pinn_small", snet, sth)
        with torch.inference_mode():
            add("компактная PINN 64×4, eager", state_size_mb(snet.state_dict()),
                timeit(lambda: sprd(rt, tt)), timeit(lambda: sinf(rt, tt)), mod_der=sinf)
        sf, sl = onnx_session(paths["full"]), onnx_session(paths["_predict"])
        add("компактная PINN 64×4, ONNX Runtime", os.path.getsize(paths["full"]) / 2 ** 20,
            timeit(lambda: sl.run(None, feeds)), timeit(lambda: sf.run(None, feeds)))

    # потоки
    for nt in (1, os.cpu_count() or 1):
        torch.set_num_threads(nt)
        with torch.inference_mode():
            add(f"eager fp32, {nt} поток(а)", size32, timeit(lambda: prd(rt, tt)), timeit(lambda: inf(rt, tt)))

    df = pd.DataFrame(rows)
    df.to_csv(f"{A}/tables/compression.csv", index=False)
    with open(f"{A}/tables/compression.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4g"))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
