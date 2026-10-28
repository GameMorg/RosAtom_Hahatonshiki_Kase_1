"""Бейзлайн: кусочно-билинейная интерполяция таблиц по областям.

Для каждой из 20 областей строится билинейный интерполятор в *нативных*
координатах сетки: если ось равномерна логарифмически — интерполяция идёт по
ln(·), если линейно — по самой величине. Так в каждой ячейке интерполяция
выполняется по равномерной сетке, как это и задумано авторами таблиц.

Режим ``values="raw"`` интерполирует P и E как есть (классический вариант),
``values="log"`` — ln P и E (точнее для P, меняющегося на порядки).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .data import RHO_EDGES, T_EDGES, Region, split_masks


def _fwd(v, scale):
    return np.log(v) if scale == "log" else v


class BilinearEOS:
    """Кусочно-билинейный интерполятор P(ρ,T), E(ρ,T) по 20 областям.

    Args:
        regions: список областей.
        use: ``"train"`` — строить только по train-узлам (для честной оценки),
            ``"all"`` — по всем узлам.
        values: ``"raw"`` или ``"log"`` (см. описание модуля).
    """

    def __init__(self, regions: list[Region], use: str = "train", values: str = "raw"):
        self.values = values
        self.items = {}
        for reg in regions:
            if use == "train":
                m = split_masks(reg)["train"]
                ii = np.where(m.any(1))[0]; jj = np.where(m.any(0))[0]
            else:
                ii = np.arange(len(reg.rho)); jj = np.arange(len(reg.T))
            r, T = reg.rho[ii], reg.T[jj]
            P, E = reg.P[np.ix_(ii, jj)], reg.E[np.ix_(ii, jj)]
            Pv = np.log(P) if values == "log" else P
            axes = (_fwd(r, reg.rho_scale), _fwd(T, reg.T_scale))
            self.items[reg.idx] = (reg.rho_scale, reg.T_scale,
                                   RegularGridInterpolator(axes, np.stack([Pv, E], -1),
                                                           bounds_error=False, fill_value=None))

    @staticmethod
    def region_index(rho, T) -> np.ndarray:
        """Номер области (1..20) для каждой точки.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            np.ndarray[int]: номер области ``5*i_rho + i_T + 1``, форма (N,).
        """
        i = np.clip(np.searchsorted(RHO_EDGES, rho, side="right") - 1, 0, 3)
        j = np.clip(np.searchsorted(T_EDGES, T, side="right") - 1, 0, 4)
        return 5 * i + j + 1

    def predict(self, rho, T):
        """Интерполяция давления и энергии.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            tuple[np.ndarray, np.ndarray]: (P [ГПа], E [кДж/г]), форма (N,).

        Raises:
            ValueError: если rho или T <= 0.
        """
        rho = np.asarray(rho, float); T = np.asarray(T, float)
        if np.any(rho <= 0) or np.any(T <= 0):
            raise ValueError("rho и T должны быть > 0")
        reg = self.region_index(rho, T)
        out = np.empty((len(rho), 2))
        for k in np.unique(reg):
            m = reg == k
            sr, sT, f = self.items[int(k)]
            out[m] = f(np.stack([_fwd(rho[m], sr), _fwd(T[m], sT)], -1))
        P = np.exp(out[:, 0]) if self.values == "log" else out[:, 0]
        return P, out[:, 1]

    def derivatives(self, rho, T, h: float = 1e-6):
        """Частные производные центральными разностями (по ln ρ, ln T).

        Returns:
            dict: P, E, dPdr, dPdT, dEdr, dEdT.
        """
        P, E = self.predict(rho, T)
        Pr1, Er1 = self.predict(rho * (1 + h), T); Pr0, Er0 = self.predict(rho * (1 - h), T)
        PT1, ET1 = self.predict(rho, T * (1 + h)); PT0, ET0 = self.predict(rho, T * (1 - h))
        return dict(P=P, E=E,
                    dPdr=(Pr1 - Pr0) / (2 * h * rho), dEdr=(Er1 - Er0) / (2 * h * rho),
                    dPdT=(PT1 - PT0) / (2 * h * T), dEdT=(ET1 - ET0) / (2 * h * T))
