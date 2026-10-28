"""Обучение Vanilla-сети и PINN.

Функция потерь:

    L = L_data + w(t) · [λ_mech·L_mech + λ_therm·L_therm + λ_sound·L_sound + λ_M·L_Maxwell]

* L_data — MSE в нормированных координатах (ln P, asinh E) на train-узлах;
* физические слагаемые (см. ``eos.model.physics_losses``) вычисляются на
  объединении train-батча и случайных коллокационных точек, равномерных по
  ln ρ × ln T во всей области (в т.ч. между узлами и на стыках областей);
* w(t) — линейный «разогрев» физики в начале обучения (сначала сеть
  выучивает грубую форму поверхности, затем включаются ограничения).

Vanilla = все λ = 0.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import Normalizer
from .model import EOSNet, Thermo, fundamental_derivative, physics_losses


@dataclass
class TrainConfig:
    """Гиперпараметры обучения."""

    name: str = "vanilla"
    width: int = 128
    depth: int = 5
    act: str = "silu"
    fourier: int = 0
    fourier_scale: float = 2.0
    epochs: int = 300
    batch: int = 4096
    lr: float = 2e-3
    lr_min: float = 1e-5
    weight_decay: float = 0.0
    lam_mech: float = 0.0
    lam_therm: float = 0.0
    lam_sound: float = 0.0
    lam_maxwell: float = 0.0
    lam_conv: float = 0.0  # штраф за 𝒢 < 0 (выпуклость изоэнтроп)
    deriv: str = "jac"  # способ производных в физ. потерях: "jac" (явный якобиан) или "autograd"
    n_colloc: int = 4096
    warmup_frac: float = 0.1
    seed: int = 0
    threads: int = 0
    init_from: str = ""  # путь к чекпойнту для «тёплого старта» (дообучение)

    @property
    def physics(self) -> bool:
        """Есть ли физические слагаемые."""
        return any(v > 0 for v in (self.lam_mech, self.lam_therm, self.lam_sound, self.lam_maxwell, self.lam_conv))


def load_processed(path: str = "data/processed/dataset.npz") -> dict:
    """Загружает train/val/test, сохранённые модулем 1."""
    z = np.load(path)
    out = {}
    for k in z.files:
        s, c = k.split("_", 1)
        out.setdefault(s, {})[c] = z[k]
    return out


def build(cfg: TrainConfig, norm: Normalizer):
    """Создаёт сеть и модуль термодинамики.

    Args:
        cfg: конфигурация обучения.
        norm: нормировщик.

    Returns:
        tuple[EOSNet, Thermo]: сеть и модуль перевода её выходов в физику.
    """
    torch.manual_seed(cfg.seed)
    net = EOSNet(cfg.width, cfg.depth, cfg.act, cfg.fourier, cfg.fourier_scale, seed=cfg.seed)
    return net, Thermo(norm)


def _x_to_phys(th: Thermo, x):
    return (torch.exp(x[:, 0] * th.lnrho_sd + th.lnrho_mu), torch.exp(x[:, 1] * th.lnT_sd + th.lnT_mu))


def train(cfg: TrainConfig, ds: dict, norm: Normalizer, log_every: int = 10, out_prefix: str | None = None):
    """Обучает модель.

    Args:
        cfg: конфигурация.
        ds: словарь train/val/test из ``load_processed``.
        norm: нормировщик.
        log_every: период (эпох) расчёта валидационных метрик.
        out_prefix: если задан — сохраняет ``{prefix}.pt`` и ``{prefix}_history.csv``.

    Returns:
        (net, thermo, history): обученная сеть, модуль термодинамики, список словарей.
    """
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    net, th = build(cfg, norm)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"])
    tr, va = ds["train"], ds["val"]
    Xtr = torch.tensor(norm.x(tr["rho"], tr["T"]), dtype=torch.float32)
    Ytr = torch.tensor(norm.y(tr["P"], tr["E"]), dtype=torch.float32)
    g = np.random.default_rng(cfg.seed)
    iv = g.choice(len(va["rho"]), size=min(60_000, len(va["rho"])), replace=False)
    Xva = torch.tensor(norm.x(va["rho"][iv], va["T"][iv]), dtype=torch.float32)
    Pva, Eva = va["P"][iv], va["E"][iv]

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(Xtr) / cfg.batch)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * min(s / total, 1)))) / cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    hist, step, t0 = [], 0, time.time()
    for ep in range(1, cfg.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        acc = {k: 0.0 for k in ("loss", "data", "mech", "therm", "sound", "maxwell", "conv")}
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch:(b + 1) * cfg.batch]
            xb, yb = Xtr[idx], Ytr[idx]
            w = min(1.0, step / max(1, cfg.warmup_frac * total))
            if cfg.physics:
                xc = torch.rand(cfg.n_colloc, 2, generator=gen) * 2 - 1
                xa = torch.cat([xb, xc])
                if cfg.deriv == "autograd":  # производные через torch.autograd (create_graph=True)
                    xa = xa.detach().requires_grad_(True)
                    y = net(xa)
                    J = torch.stack([torch.autograd.grad(y[:, i].sum(), xa, create_graph=True)[0] for i in range(2)], 1)
                else:  # эквивалентный явный якобиан (быстрее, совпадает с autograd)
                    y, J = net.forward_jac(xa)
                ld = (y[: len(xb)] - yb).pow(2).mean()
                rho, T = _x_to_phys(th, xa)
                pl = physics_losses(th.all(rho, T, y, J))
                lp = (cfg.lam_mech * pl["mech"] + cfg.lam_therm * pl["therm"]
                      + cfg.lam_sound * pl["sound"] + cfg.lam_maxwell * pl["maxwell"])
                if cfg.lam_conv > 0:
                    G, pos = fundamental_derivative(net, th, xc[: cfg.n_colloc // 2])
                    lconv = (torch.relu(0.05 - G).clamp(max=5.0) * pos).mean()
                    lp = lp + cfg.lam_conv * lconv
                    pl["conv"] = lconv
                loss = ld + w * lp
            else:
                ld = (net(xb) - yb).pow(2).mean(); pl = {}; loss = ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            acc["loss"] += loss.item(); acc["data"] += ld.item()
            for k, v in pl.items():
                acc[k] += v.item()
        rec = {k: v / steps_per_epoch for k, v in acc.items()}
        rec.update(epoch=ep, lr=sched.get_last_lr()[0], time_s=time.time() - t0)
        if ep % log_every == 0 or ep == cfg.epochs or ep == 1:
            net.eval()
            with torch.no_grad():
                P, E = norm.inv_y(net(Xva).double().numpy())
            rec["val_P_mape_pct"] = float(100 * np.mean(np.abs(P - Pva) / Pva))
            rec["val_E_mre_pct"] = float(100 * np.mean(np.abs(E - Eva) / np.maximum(np.abs(Eva), 1.0)))
            print(f"[{cfg.name}] ep {ep:4d} loss {rec['loss']:.3e} data {rec['data']:.3e} "
                  f"mech {rec['mech']:.1e} therm {rec['therm']:.1e} sound {rec['sound']:.1e} "
                  f"maxw {rec['maxwell']:.2e} conv {rec['conv']:.1e} | val P {rec['val_P_mape_pct']:.3f}% "
                  f"E {rec['val_E_mre_pct']:.3f}% | {rec['time_s']:.0f}s", flush=True)
        hist.append(rec)
    net.eval()
    if out_prefix:
        save_checkpoint(f"{out_prefix}.pt", net, cfg, norm)
        import pandas as pd
        pd.DataFrame(hist).to_csv(f"{out_prefix}_history.csv", index=False)
    return net, th, hist


def save_checkpoint(path: str, net: EOSNet, cfg: TrainConfig, norm: Normalizer) -> None:
    """Сохраняет state_dict, конфигурацию и нормировку в один .pt.

    Args:
        path: путь к файлу.
        net: обученная сеть.
        cfg: конфигурация обучения.
        norm: нормировщик.
    """
    torch.save({"state_dict": net.state_dict(), "config": asdict(cfg), "norm": norm.__dict__}, path)


def load_checkpoint(path: str):
    """Загружает (net, thermo, cfg, norm) из .pt-чекпойнта."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"]); norm = Normalizer(**ck["norm"])
    net, th = build(cfg, norm)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, th, cfg, norm


def config_from_json(path: str) -> TrainConfig:
    """Читает TrainConfig из JSON."""
    with open(path, encoding="utf-8") as f:
        return TrainConfig(**json.load(f))
