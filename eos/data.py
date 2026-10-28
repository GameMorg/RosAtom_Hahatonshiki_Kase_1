"""Загрузка табличного УРС алюминия (20 областей), разбиение и нормирование.

Каждая область ``OblNN.txt`` — прямоугольная сетка (ρ, T), равномерная по каждой
оси линейно (``lin``) или логарифмически (``log``). Всего 4 типа сеток:
log-log, log-lin, lin-log, lin-lin. Области стыкуются по границам:

    ρ:  A [1e-3, 0.1]   B [0.1, 1.5]   C [1.5, 5]   D [5, 1e4]      г/см³
    T:  1 [0.1, 1e2]    2 [1e2, 3e3]   3 [3e3, 3e4] 4 [3e4, 1e5] 5 [1e5, 1e7]  K

Номер области: ``Obl = 5*i_rho + i_T + 1``.

Единицы: ρ [г/см³], T [K], P [ГПа], E [кДж/г]. Заметим, что
ГПа / (г/см³) = кДж/г, поэтому соотношения вида ρ²·∂E/∂ρ = P − T·∂P/∂T
записываются без множителей пересчёта.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from glob import glob

import numpy as np

RHO_EDGES = [1e-3, 0.1, 1.5, 5.0, 1e4]
T_EDGES = [0.1, 1e2, 3e3, 3e4, 1e5, 1e7]
RHO_MIN, RHO_MAX = RHO_EDGES[0], RHO_EDGES[-1]
T_MIN, T_MAX = T_EDGES[0], T_EDGES[-1]


@dataclass
class Region:
    """Одна табличная область (прямоугольная сетка).

    Attributes:
        name: имя области, например ``"Obl07"``.
        rho: узлы по плотности, г/см³, форма (Nr,).
        T: узлы по температуре, K, форма (NT,).
        P: давление, ГПа, форма (Nr, NT).
        E: удельная внутренняя энергия, кДж/г, форма (Nr, NT).
        rho_scale: ``"lin"`` или ``"log"`` — тип равномерности сетки по ρ.
        T_scale: ``"lin"`` или ``"log"`` — тип равномерности сетки по T.
    """

    name: str
    rho: np.ndarray
    T: np.ndarray
    P: np.ndarray
    E: np.ndarray
    rho_scale: str
    T_scale: str

    @property
    def grid_type(self) -> str:
        """Тип сетки в виде ``"log-lin"`` (ρ-T)."""
        return f"{self.rho_scale}-{self.T_scale}"

    @property
    def idx(self) -> int:
        """Номер области (1..20)."""
        return int(self.name[3:])

    def flat(self, mask: np.ndarray | None = None) -> dict:
        """Плоское представление узлов области.

        Args:
            mask: булева маска формы (Nr, NT) или None (все узлы).

        Returns:
            dict[str, np.ndarray]: ``rho, T, P, E, region`` формы (M,).
        """
        R, TT = np.meshgrid(self.rho, self.T, indexing="ij")
        m = np.ones_like(R, bool) if mask is None else mask
        return dict(rho=R[m], T=TT[m], P=self.P[m], E=self.E[m],
                    region=np.full(m.sum(), self.idx, np.int16))


def _parse_header(path: str) -> dict:
    hdr = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"Grid type:\s*rho=(\w+),\s*T=(\w+)", line)
            if m:
                hdr["rho_scale"], hdr["T_scale"] = m.group(1), m.group(2)
    return hdr


def load_region(path: str) -> Region:
    """Читает один файл области и восстанавливает 2D-сетку.

    Args:
        path: путь к ``OblNN.txt``.

    Returns:
        Region: сетка области.

    Raises:
        ValueError: если точки не образуют полную прямоугольную сетку.
    """
    hdr = _parse_header(path)
    a = np.loadtxt(path, comments="#")
    rho, T = np.unique(a[:, 0]), np.unique(a[:, 1])
    if len(rho) * len(T) != len(a):
        raise ValueError(f"{path}: точки не образуют прямоугольную сетку")
    # в файлах быстрый индекс — ρ; приводим к виду [i_rho, j_T]
    order = np.lexsort((a[:, 0], a[:, 1]))
    a = a[order]
    P = a[:, 2].reshape(len(T), len(rho)).T.copy()
    E = a[:, 3].reshape(len(T), len(rho)).T.copy()
    name = os.path.basename(path).split(".")[0]
    return Region(name, rho, T, P, E, hdr.get("rho_scale", "log"), hdr.get("T_scale", "log"))


def load_all(data_dir: str) -> list[Region]:
    """Загружает все 20 областей из каталога ``data_dir`` (файлы ``Obl*.txt``)."""
    files = sorted(glob(os.path.join(data_dir, "Obl*.txt")))
    if not files:
        raise FileNotFoundError(f"В {data_dir} нет файлов Obl*.txt")
    return [load_region(f) for f in files]


def split_masks(reg: Region) -> dict[str, np.ndarray]:
    """Детерминированное разбиение узлов сетки.

    * ``train`` — узлы (чётный i, чётный j): «прореженная» в 2 раза сетка;
    * ``test``  — узлы (нечётный i, нечётный j): центры ячеек train-сетки,
      самые трудные точки для интерполяции;
    * ``val``   — остальные (середины рёбер).

    Такое разбиение даёт честное сравнение: билинейный бейзлайн строится только
    по train-узлам, и нейросеть видит только их же.
    """
    I, J = np.meshgrid(np.arange(len(reg.rho)), np.arange(len(reg.T)), indexing="ij")
    ei, ej = I % 2 == 0, J % 2 == 0
    # крайние узлы включаем в train, чтобы train-сетка покрывала всю область
    ei |= I == len(reg.rho) - 1
    ej |= J == len(reg.T) - 1
    train = ei & ej
    test = (~ei) & (~ej)
    val = ~(train | test)
    return dict(train=train, val=val, test=test)


def make_dataset(regions: list[Region]) -> dict[str, dict[str, np.ndarray]]:
    """Собирает плоские train/val/test массивы по всем областям."""
    out = {k: [] for k in ("train", "val", "test")}
    for reg in regions:
        for k, m in split_masks(reg).items():
            out[k].append(reg.flat(m))
    return {k: {c: np.concatenate([d[c] for d in v]) for c in v[0]} for k, v in out.items()}


@dataclass
class Normalizer:
    """Нормировка входов и выходов.

    Входы:  x_ρ = (ln ρ − ρ_mu)/ρ_sd,  x_T = (ln T − T_mu)/T_sd.
    Выходы: y_P = (ln P − P_mu)/P_sd  (P > 0 во всей области),
            y_E = (asinh(E/E_s) − E_mu)/E_sd.

    asinh ведёт себя как логарифм при |E| ≫ E_s и линейно около нуля,
    поэтому корректно обрабатывает смену знака E.
    """

    lnrho_mu: float = 0.0
    lnrho_sd: float = 1.0
    lnT_mu: float = 0.0
    lnT_sd: float = 1.0
    lnP_mu: float = 0.0
    lnP_sd: float = 1.0
    E_s: float = 1.0
    aE_mu: float = 0.0
    aE_sd: float = 1.0
    units: dict = field(default_factory=lambda: dict(rho="g/cm^3", T="K", P="GPa", E="kJ/g"))
    domain: dict = field(default_factory=lambda: dict(rho=[RHO_MIN, RHO_MAX], T=[T_MIN, T_MAX]))

    @classmethod
    def fit(cls, rho, T, P, E, E_s: float = 1.0) -> "Normalizer":
        """Оценивает параметры нормировки по обучающей выборке.

        Для входов используются границы всей области (а не выборочные моменты),
        поэтому x ∈ [-1, 1] независимо от плотности сетки.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).
            P: давление, ГПа, форма (N,).
            E: энергия, кДж/г, форма (N,).
            E_s: масштаб asinh-преобразования энергии, кДж/г.

        Returns:
            Normalizer: параметры нормировки.
        """
        lr = np.log([RHO_MIN, RHO_MAX]); lt = np.log([T_MIN, T_MAX])
        lP = np.log(P); aE = np.arcsinh(E / E_s)
        return cls(lnrho_mu=float(lr.mean()), lnrho_sd=float((lr[1] - lr[0]) / 2),
                   lnT_mu=float(lt.mean()), lnT_sd=float((lt[1] - lt[0]) / 2),
                   lnP_mu=float(lP.mean()), lnP_sd=float(lP.std()),
                   E_s=float(E_s), aE_mu=float(aE.mean()), aE_sd=float(aE.std()))

    # --- прямое/обратное преобразование (numpy) ---
    def x(self, rho, T) -> np.ndarray:
        """Нормированные входы сети.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            np.ndarray: x = ((ln ρ − μ)/σ, (ln T − μ)/σ), форма (N, 2).
        """
        return np.stack([(np.log(rho) - self.lnrho_mu) / self.lnrho_sd,
                         (np.log(T) - self.lnT_mu) / self.lnT_sd], -1)

    def y(self, P, E) -> np.ndarray:
        """Нормированные целевые значения.

        Args:
            P: давление, ГПа, форма (N,).
            E: энергия, кДж/г, форма (N,).

        Returns:
            np.ndarray: (y_P, y_E), форма (N, 2).
        """
        return np.stack([(np.log(P) - self.lnP_mu) / self.lnP_sd,
                         (np.arcsinh(E / self.E_s) - self.aE_mu) / self.aE_sd], -1)

    def inv_y(self, y) -> tuple[np.ndarray, np.ndarray]:
        """Обратное преобразование выходов сети.

        Args:
            y: нормированные выходы, форма (..., 2).

        Returns:
            tuple[np.ndarray, np.ndarray]: (P [ГПа], E [кДж/г]).
        """
        P = np.exp(y[..., 0] * self.lnP_sd + self.lnP_mu)
        E = self.E_s * np.sinh(y[..., 1] * self.aE_sd + self.aE_mu)
        return P, E

    def to_json(self, path: str) -> None:
        """Сохраняет коэффициенты нормирования в JSON.

        Args:
            path: путь к файлу ``normalization.json``.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "Normalizer":
        """Загружает коэффициенты нормирования из JSON.

        Args:
            path: путь к файлу ``normalization.json``.

        Returns:
            Normalizer: параметры нормировки.
        """
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
