"""Нейросетевая модель УРС и термодинамические величины (PyTorch).

Сеть ``EOSNet`` отображает нормированные входы x = (x_ρ, x_T) ∈ [-1, 1]² в
нормированные выходы y = (y_P, y_E) (см. ``eos.data.Normalizer``):

    y_P = (ln P − μ_P)/σ_P,     y_E = (asinh(E/E_s) − μ_E)/σ_E.

Помимо обычного ``forward`` сеть умеет за один проход считать якобиан
J = ∂y/∂x (forward-mode, «касательные» протаскиваются через слои явно).
Это (а) в ~2–3 раза дешевле, чем autograd с ``create_graph=True``, (б) остаётся
дифференцируемым по весам — используется в физической функции потерь,
(в) экспортируется в TorchScript/ONNX, так что производные и скорость звука
доступны в инференсе и в ONNX. Эталонный расчёт производных — ``autograd_derivatives``
(torch.autograd, п. 2.3 ТЗ); совпадение двух способов проверяется тестами.
"""
from __future__ import annotations

import math

import torch
from torch import nn

C_LIGHT_SQ = 299_792_458.0 ** 2 / 1e6  # кДж/г


def _act(z: torch.Tensor, kind: str):
    """Активация и её производная."""
    if kind == "tanh":
        a = torch.tanh(z)
        return a, 1 - a * a
    s = torch.sigmoid(z)  # SiLU
    return z * s, s * (1 + z * (1 - s))


class EOSNet(nn.Module):
    """MLP 2 → (width × depth) → 2 с гладкой активацией.

    Args:
        width: число нейронов в скрытом слое.
        depth: число скрытых слоёв.
        act: ``"silu"`` или ``"tanh"`` (нужна гладкость: в функции потерь
            участвуют производные выходов по входам).
        fourier: число частот случайных Фурье-признаков (0 — без них).
            Признаки sin/cos(πBx) помогают описывать резкие переходы
            (ионизация, «холодная» кривая около нормальной плотности).
    """

    def __init__(self, width: int = 128, depth: int = 5, act: str = "silu", fourier: int = 0,
                 fourier_scale: float = 2.0, seed: int = 0):
        super().__init__()
        self.act = act
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, fourier, generator=g) * fourier_scale)
        d_in = 2 + 2 * fourier
        dims = [d_in] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(width, 2)
        for l in self.hidden:
            nn.init.xavier_uniform_(l.weight, gain=1.0 if act == "tanh" else math.sqrt(2))
            nn.init.zeros_(l.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _features(self, x):
        if self.B.shape[1] == 0:
            return x, None
        z = math.pi * x @ self.B
        f = torch.cat([x, torch.sin(z), torch.cos(z)], -1)
        # ∂f/∂x: (N, d_in, 2)
        dz = math.pi * self.B.T  # (F, 2)
        df = torch.cat([torch.eye(2, device=x.device, dtype=x.dtype).expand(x.shape[0], 2, 2),
                        torch.cos(z)[..., None] * dz, -torch.sin(z)[..., None] * dz], 1)
        return f, df

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Нормированные выходы y, форма (N, 2)."""
        h, _ = self._features(x)
        for l in self.hidden:
            h, _ = _act(l(h), self.act)
        return self.out(h)

    def forward_jac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Выходы и якобиан.

        Касательные по двум входам хранятся как два «дополнительных батча»
        (тензор формы (3N, width): значения + 2 касательных), поэтому каждый слой —
        одно матричное умножение, без einsum по батчу.

        Returns:
            y: (N, 2); J: (N, 2, 2), J[:, i, k] = ∂y_i/∂x_k.
        """
        n = x.shape[0]
        h, dh = self._features(x)
        if dh is None:
            eye = torch.eye(2, device=x.device, dtype=x.dtype)
            d0, d1 = eye[0].expand(n, 2), eye[1].expand(n, 2)
        else:  # dh: (N, d_in, 2)
            d0, d1 = dh[..., 0], dh[..., 1]
        for l in self.hidden:
            z = l(h)
            dz = torch.cat([d0, d1], 0) @ l.weight.T  # (2N, width)
            h, da = _act(z, self.act)
            d = dz * da.repeat(2, 1)
            d0, d1 = d[:n], d[n:]
        y = self.out(h)
        j0, j1 = d0 @ self.out.weight.T, d1 @ self.out.weight.T  # (N, 2): ∂y/∂x_0, ∂y/∂x_1
        return y, torch.stack([j0, j1], -1)


class Thermo(nn.Module):
    """Перевод (y, J) сети в физические величины и производные.

    Хранит коэффициенты нормирования как буферы — модуль самодостаточен и
    экспортируется вместе с сетью.
    """

    def __init__(self, norm):
        super().__init__()
        for k in ("lnrho_mu", "lnrho_sd", "lnT_mu", "lnT_sd", "lnP_mu", "lnP_sd", "E_s", "aE_mu", "aE_sd"):
            self.register_buffer(k, torch.tensor(float(getattr(norm, k))))

    def x(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """(ρ, T) → нормированные входы сети.

        Args:
            rho: плотность, г/см³, тензор (N,).
            T: температура, K, тензор (N,).

        Returns:
            torch.Tensor: x, форма (N, 2).
        """
        return torch.stack([(torch.log(rho) - self.lnrho_mu) / self.lnrho_sd,
                            (torch.log(T) - self.lnT_mu) / self.lnT_sd], -1)

    def values(self, y: torch.Tensor):
        """Выходы сети → физические значения.

        Args:
            y: нормированные выходы сети, форма (N, 2).

        Returns:
            tuple: (P [ГПа], E [кДж/г], ln P, asinh(E/E_s)), каждый формы (N,).
        """
        lnP = y[:, 0] * self.lnP_sd + self.lnP_mu
        aE = y[:, 1] * self.aE_sd + self.aE_mu
        return torch.exp(lnP), self.E_s * 0.5 * (torch.exp(aE) - torch.exp(-aE)), lnP, aE  # sinh (ONNX-совместимо)

    def all(self, rho, T, y, J) -> dict:
        """Все физические величины по выходам и якобиану сети.

        Обозначения безразмерных величин:
            g_r = ∂lnP/∂lnρ,  g_T = ∂lnP/∂lnT,
            e_r = ∂E/∂lnρ,    e_T = ∂E/∂lnT = T·C_v.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).
            y: выходы сети, форма (N, 2).
            J: якобиан ∂y/∂x, форма (N, 2, 2).

        Returns:
            dict[str, torch.Tensor]: P, E, dPdr, dPdT, dEdr, dEdT, cs2 (км²/с²), cs2n, rM (невязка
            Максвелла), P_over_rho, g_r, g_T, e_T, J.
        """
        P, E, lnP, aE = self.values(y)
        coshE = self.E_s * 0.5 * (torch.exp(aE) + torch.exp(-aE))  # cosh (ONNX-совместимо)
        g_r = J[:, 0, 0] * self.lnP_sd / self.lnrho_sd
        g_T = J[:, 0, 1] * self.lnP_sd / self.lnT_sd
        e_r = coshE * J[:, 1, 0] * self.aE_sd / self.lnrho_sd
        e_T = coshE * J[:, 1, 1] * self.aE_sd / self.lnT_sd
        dPdr, dPdT = P * g_r / rho, P * g_T / T
        dEdr, dEdT = e_r / rho, e_T / T
        # c_s² = ∂P/∂ρ + T/(ρ²C_v)(∂P/∂T)² = (P/ρ)·[g_r + P g_T² / (ρ e_T)]
        cs2n = g_r + P * g_T ** 2 / (rho * e_T)  # безразмерная c_s²·ρ/P
        cs2 = P / rho * cs2n
        # невязка Максвелла: (ρ²∂E/∂ρ − P + T∂P/∂T)/P = ρ e_r / P − 1 + g_T
        rM = rho * e_r / P - 1 + g_T
        return dict(P=P, E=E, dPdr=dPdr, dPdT=dPdT, dEdr=dEdr, dEdT=dEdT, cs2=cs2, cs2n=cs2n, rM=rM, P_over_rho=P / rho,
                    g_r=g_r, g_T=g_T, e_T=e_T, J=J)


def physics_losses(th: dict, margin: float = 1e-5) -> dict[str, torch.Tensor]:
    """Слагаемые физической функции потерь (все безразмерные).

    * ``mech``   — механическая устойчивость: линейный hinge relu(margin − ∂lnP/∂lnρ);
    * ``therm``  — термическая устойчивость: relu(margin − ∂y_E/∂x_T) (⇔ C_v < 0).
      Линейный (а не квадратичный) hinge выбран намеренно: нарушения C_v < 0 у Vanilla
      многочисленны, но малы, и квадратичный штраф почти не даёт для них градиента;
    * ``sound``  — причинность c_s < c: relu(ln(c_s²/c²))² только в точках с C_v > 0
      (прямой штраф за c_s² < 0 при C_v → 0⁻ даёт взрывные градиенты; положительность
      c_s² следует из условий mech и therm);
    * ``maxwell``— термодинамическая согласованность: Huber(r_M, δ = 0.5).

    Args:
        th: словарь величин из ``Thermo.all``.
        margin: малый запас ε в hinge-штрафах.

    Returns:
        dict[str, torch.Tensor]: скалярные слагаемые ``mech, therm, sound, maxwell``.
    """
    J = th["J"]
    mech = torch.relu(margin - th["g_r"]).mean()
    therm = torch.relu(margin - J[:, 1, 1]).mean()
    Pr = th["P_over_rho"]
    pos = th["e_T"] > 0  # точки с C_v ≤ 0 уже штрафуются слагаемым therm
    e_T = torch.where(pos, th["e_T"], torch.ones_like(th["e_T"]))
    cs2 = Pr * (th["g_r"].clamp(min=0) + Pr * th["g_T"] ** 2 / e_T)
    lc = torch.log(torch.clamp(cs2, min=1e-30) / C_LIGHT_SQ)
    sound = (torch.relu(lc).pow(2) * pos).mean()
    maxwell = torch.nn.functional.huber_loss(th["rM"], torch.zeros_like(th["rM"]), delta=0.5)
    return dict(mech=mech, therm=therm, sound=sound, maxwell=maxwell)


def fundamental_derivative(net: "EOSNet", th: "Thermo", x: torch.Tensor, h: float = 2e-3):
    """Фундаментальная производная 𝒢 (выпуклость изоэнтроп) в точках x.

    𝒢 = 1 + ½·(∂ln c_s²/∂lnρ)_S,  (∂/∂lnρ)_S = ∂/∂lnρ|_T + Γ·∂/∂lnT|_ρ,  Γ = P·g_T/(ρ·e_T).
    Производные ln c_s² берутся конечными разностями по нормированным входам
    (два дополнительных прохода сети), поэтому 𝒢 дифференцируема по весам.

    Returns:
        (G, pos): 𝒢 и маска точек с C_v > 0 и c_s² > 0 (где 𝒢 определена).
    """
    def lncs2(xx):
        y, J = net.forward_jac(xx)
        rho = torch.exp(xx[:, 0] * th.lnrho_sd + th.lnrho_mu); T = torch.exp(xx[:, 1] * th.lnT_sd + th.lnT_mu)
        d = th.all(rho, T, y, J)
        return d, torch.log(torch.clamp(d["cs2"], min=1e-30))
    d0, l0 = lncs2(x)
    _, lr = lncs2(x + torch.tensor([h, 0.0], dtype=x.dtype))
    _, lt = lncs2(x + torch.tensor([0.0, h], dtype=x.dtype))
    dl_dlnrho = (lr - l0) / (h * th.lnrho_sd)
    dl_dlnT = (lt - l0) / (h * th.lnT_sd)
    pos = (d0["e_T"] > 0) & (d0["cs2"] > 0)
    gam = d0["P_over_rho"] * d0["g_T"] / torch.where(pos, d0["e_T"], torch.ones_like(d0["e_T"]))
    G = 1 + 0.5 * (dl_dlnrho + gam * dl_dlnT)
    return G, pos


def autograd_derivatives(net: "EOSNet", th: "Thermo", rho: torch.Tensor, T: torch.Tensor,
                         create_graph: bool = False) -> dict[str, torch.Tensor]:
    """Частные производные P и E через ``torch.autograd`` (п. 2.3 ТЗ).

    Производные берутся по физическим входам ρ и T напрямую от выходов сети,
    поэтому это эталонный способ. Явный якобиан ``EOSNet.forward_jac`` даёт те же
    значения (проверено тестами и в ``scripts/m2_autograd_derivatives.py``) и нужен
    только для экспорта в ONNX/TorchScript и скорости.

    Args:
        net: сеть.
        th: модуль термодинамики (нормировка).
        rho: плотность, г/см³, тензор (N,).
        T: температура, K, тензор (N,).
        create_graph: сохранить граф (нужно, если производные входят в функцию потерь).

    Returns:
        dict[str, torch.Tensor]: P, E, dPdr, dPdT, dEdr, dEdT, cs2.
    """
    r = rho.detach().clone().requires_grad_(True)
    t = T.detach().clone().requires_grad_(True)
    P, E, _, _ = th.values(net(th.x(r, t)))
    dPdr, dPdT = torch.autograd.grad(P.sum(), (r, t), create_graph=True)
    dEdr, dEdT = torch.autograd.grad(E.sum(), (r, t), create_graph=create_graph)
    if not create_graph:
        dPdr, dPdT = dPdr.detach(), dPdT.detach()
    cs2 = dPdr + t * dPdT ** 2 / (r ** 2 * dEdT)
    out = dict(P=P, E=E, dPdr=dPdr, dPdT=dPdT, dEdr=dEdr, dEdT=dEdT, cs2=cs2)
    return out if create_graph else {k: v.detach() for k, v in out.items()}


class EOSInference(nn.Module):
    """Экспортируемый модуль: (ρ, T) → [P, E, ∂P/∂ρ, ∂P/∂T, ∂E/∂ρ, ∂E/∂T, c_s²].

    Вход — два тензора формы (N,) в физических единицах (float32).
    Выход — тензор (N, 7). Производные считаются явным якобианом (эквивалент autograd),
    поэтому модуль работает в TorchScript и ONNX Runtime.
    """

    def __init__(self, net: EOSNet, thermo: Thermo):
        super().__init__()
        self.net, self.thermo = net, thermo

    def forward(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Значения, производные и c_s².

        Args:
            rho: плотность, г/см³, тензор (N,) float32.
            T: температура, K, тензор (N,) float32.

        Returns:
            torch.Tensor: (N, 7) — P [ГПа], E [кДж/г], ∂P/∂ρ, ∂P/∂T, ∂E/∂ρ, ∂E/∂T (= C_v), c_s² [км²/с²].
        """
        x = self.thermo.x(rho, T)
        y, J = self.net.forward_jac(x)
        d = self.thermo.all(rho, T, y, J)
        return torch.stack([d["P"], d["E"], d["dPdr"], d["dPdT"], d["dEdr"], d["dEdT"], d["cs2"]], -1)


class EOSPredict(nn.Module):
    """Экспортируемый «лёгкий» модуль: (ρ, T) → [P, E] (только значения)."""

    def __init__(self, net: EOSNet, thermo: Thermo):
        super().__init__()
        self.net, self.thermo = net, thermo

    def forward(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Только значения.

        Args:
            rho: плотность, г/см³, тензор (N,).
            T: температура, K, тензор (N,).

        Returns:
            torch.Tensor: (N, 2) — P [ГПа], E [кДж/г].
        """
        P, E, _, _ = self.thermo.values(self.net(self.thermo.x(rho, T)))
        return torch.stack([P, E], -1)
