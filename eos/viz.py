"""Единый стиль графиков (matplotlib)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# последовательная шкала: один тон (синий), светлый → тёмный
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc3ee", "#2a78d6", "#123e73"])
# расходящаяся: синий ↔ нейтральный серый ↔ оранжевый
DIV = LinearSegmentedColormap.from_list("div", ["#1d5fae", "#8fb6e6", "#e9e8e4", "#f2a17e", "#b8461a"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "lines.linewidth": 1.6, "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def save(fig, path: str) -> None:
    """Сохраняет фигуру в файл и закрывает её.

    Args:
        fig: фигура matplotlib.
        path: путь к PNG.
    """
    fig.savefig(path)
    plt.close(fig)
