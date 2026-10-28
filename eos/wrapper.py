"""Класс-обёртка для использования обученного УРС в расчётах.

Пример::

    from eos.wrapper import EOSModel
    eos = EOSModel("artifacts/models/pinn.onnx")          # или .pt / .ts.pt
    P, E = eos.predict(rho=[2.7, 10.0], T=[300.0, 1e4])    # ГПа, кДж/г
    cs = eos.sound_speed([2.7], [300.0])                  # км/с
    flags = eos.check_physics([2.7], [300.0])              # dict булевых массивов
"""
from __future__ import annotations

import warnings

import numpy as np

from . import physics as ph
from .data import RHO_MAX, RHO_MIN, T_MAX, T_MIN

_COLS = ("P", "E", "dPdr", "dPdT", "dEdr", "dEdT", "cs2")


class EOSModel:
    """Нейросетевое уравнение состояния алюминия.

    Args:
        model_path: путь к модели. Поддерживаются:
            ``*.onnx`` (ONNX Runtime, torch не нужен),
            ``*.ts.pt`` (TorchScript-модуль ``EOSInference``),
            ``*.pt`` (чекпойнт: state_dict + config + нормировка).
        batch_size: размер батча при инференсе.
        ood: поведение вне области обучения: ``"warn"`` (по умолчанию — значение
            экстраполяции + RuntimeWarning) или ``"nan"`` (вернуть NaN, как у таблиц).
            Вне области сеть не гарантирует ни точности, ни физики: например,
            при ρ > 10⁴ г/см³ давление начинает падать с ростом ρ.

    Raises:
        ValueError: неизвестный формат файла.
    """

    def __init__(self, model_path: str, batch_size: int = 200_000, ood: str = "warn"):
        if ood not in ("warn", "nan"):
            raise ValueError('ood должен быть "warn" или "nan"')
        self.path, self.batch_size, self.ood = model_path, batch_size, ood
        if model_path.endswith(".onnx"):
            import onnxruntime as ort
            so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
            self._run = self._run_onnx
        elif model_path.endswith(".pt"):
            import torch
            self._torch = torch
            if model_path.endswith(".ts.pt"):
                self._mod = torch.jit.load(model_path, map_location="cpu")
            else:
                from .model import EOSInference
                from .train import load_checkpoint
                net, th, _, _ = load_checkpoint(model_path)
                self._mod = EOSInference(net, th)
            self._mod.eval()
            self._run = self._run_torch
        else:
            raise ValueError(f"Неизвестный формат модели: {model_path}")

    # --- бэкенды ---
    def _run_onnx(self, rho, T):
        return self._sess.run(None, {"rho": rho.astype(np.float32), "T": T.astype(np.float32)})[0]

    def _run_torch(self, rho, T):
        t = self._torch
        with t.inference_mode():
            return self._mod(t.from_numpy(rho.astype(np.float32)), t.from_numpy(T.astype(np.float32))).numpy()

    # --- служебное ---
    @staticmethod
    def _validate(rho, T):
        rho = np.atleast_1d(np.asarray(rho, dtype=np.float64)).ravel()
        T = np.atleast_1d(np.asarray(T, dtype=np.float64)).ravel()
        if rho.shape != T.shape:
            raise ValueError(f"rho и T должны иметь одинаковую форму: {rho.shape} vs {T.shape}")
        if np.any(~np.isfinite(rho)) or np.any(~np.isfinite(T)) or np.any(rho <= 0) or np.any(T <= 0):
            raise ValueError("rho и T должны быть конечными и > 0")
        out = (rho < RHO_MIN) | (rho > RHO_MAX) | (T < T_MIN) | (T > T_MAX)
        if out.any():
            warnings.warn(f"{out.sum()} точек вне области обучения "
                          f"(ρ∈[{RHO_MIN:g},{RHO_MAX:g}] г/см³, T∈[{T_MIN:g},{T_MAX:g}] K): экстраполяция",
                          RuntimeWarning, stacklevel=3)
        return rho, T

    def _all(self, rho, T) -> np.ndarray:
        rho, T = self._validate(rho, T)
        outs = [self._run(rho[i:i + self.batch_size], T[i:i + self.batch_size])
                for i in range(0, len(rho), self.batch_size)]
        o = np.concatenate(outs).astype(np.float64)
        if self.ood == "nan":
            o[(rho < RHO_MIN) | (rho > RHO_MAX) | (T < T_MIN) | (T > T_MAX)] = np.nan
        return rho, T, o

    # --- публичный API ---
    def predict(self, rho, T) -> tuple[np.ndarray, np.ndarray]:
        """Предсказание давления и энергии.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            (P, E): давление (ГПа) и удельная внутренняя энергия (кДж/г), форма (N,).

        Raises:
            ValueError: если rho или T <= 0, не конечны или разной формы.
        """
        _, _, o = self._all(rho, T)
        return o[:, 0], o[:, 1]

    def derivatives(self, rho, T) -> dict[str, np.ndarray]:
        """Значения и частные производные.

        Returns:
            dict: ``P, E`` и ``dPdr`` = (∂P/∂ρ)_T [ГПа·см³/г], ``dPdT`` = (∂P/∂T)_ρ [ГПа/K],
            ``dEdr`` = (∂E/∂ρ)_T [кДж·см³/г²], ``dEdT`` = C_v [кДж/(г·K)], ``cs2`` [км²/с²].
        """
        _, _, o = self._all(rho, T)
        return {k: o[:, i] for i, k in enumerate(_COLS)}

    def derivatives_autograd(self, rho, T) -> dict[str, np.ndarray]:
        """Частные производные через ``torch.autograd`` (эталон, п. 2.3 ТЗ).

        Доступно для чекпойнта ``.pt`` (нужен PyTorch). Значения совпадают с
        ``derivatives`` (явный якобиан) до точности float32.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            dict[str, np.ndarray]: P, E, dPdr, dPdT, dEdr, dEdT, cs2.

        Raises:
            ValueError: если модель загружена не из ``.pt``-чекпойнта или вход некорректен.
        """
        if not (self.path.endswith(".pt") and not self.path.endswith(".ts.pt")):
            raise ValueError("autograd доступен только для чекпойнта .pt (PyTorch)")
        from .model import autograd_derivatives
        rho, T = self._validate(rho, T)
        t = self._torch
        d = autograd_derivatives(self._mod.net, self._mod.thermo, t.tensor(rho, dtype=t.float32), t.tensor(T, dtype=t.float32))
        return {k: v.double().numpy() for k, v in d.items()}

    def sound_speed(self, rho, T) -> np.ndarray:
        """Адиабатическая скорость звука.

        c_s² = (∂P/∂ρ)_T + T/(ρ²·(∂E/∂T)_ρ) · (∂P/∂T)_ρ².

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            c_s, км/с, форма (N,). Если c_s² ≤ 0 (нарушение устойчивости) — NaN.
        """
        d = self.derivatives(rho, T)
        with np.errstate(invalid="ignore"):
            return np.where(d["cs2"] > 0, np.sqrt(np.abs(d["cs2"])), np.nan)

    def check_physics(self, rho, T, maxwell_tol: float = 0.05) -> dict[str, np.ndarray]:
        """Проверка термодинамических ограничений в точках.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).
            maxwell_tol: допуск на безразмерную невязку Максвелла |r_M|.

        Returns:
            dict[str, np.ndarray[bool]] — True означает НАРУШЕНИЕ:
            ``P_le0`` (P ≤ 0), ``dPdrho_le0`` ((∂P/∂ρ)_T ≤ 0), ``Cv_lt0`` (C_v < 0),
            ``cs2_le0`` (c_s² ≤ 0), ``cs_ge_c`` (c_s ≥ c),
            ``maxwell`` (|r_M| > maxwell_tol), ``any_stability`` (любое из первых
            четырёх условий устойчивости), ``out_of_domain`` (точка вне области обучения
            ρ ∈ [1e-3, 1e4] г/см³, T ∈ [0.1, 1e7] K: значение — экстраполяция, физика не
            гарантирована), а также ``r_maxwell`` (float, сама невязка).
        """
        rho, T, o = self._all(rho, T)
        d = {k: o[:, i] for i, k in enumerate(_COLS)}
        f = ph.violations(rho, T, d["P"], d["E"], d["dPdr"], d["dPdT"], d["dEdr"], d["dEdT"], maxwell_tol)
        f["r_maxwell"] = ph.maxwell_residual(rho, T, d["P"], d["dPdT"], d["dEdr"])
        f["out_of_domain"] = (rho < RHO_MIN) | (rho > RHO_MAX) | (T < T_MIN) | (T > T_MAX)
        return f
