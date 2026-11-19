"""One look for every figure: a fixed colour per implementation, light grid, no top/right frame."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "Dynamax": "#9b9b9b",
    "Newton example": "#9b9b9b",
    "JAX natural": "#e07b39",
    "JAX lanes": "#d9a441",
    "Pallas: kernel per step": "#7fb3d5",
    "Pallas: whole loop": "#1f5fa8",
    "Pallas: whole rollout": "#1f5fa8",
    "Warp (ours)": "#5aa469",
    "CPU": "#6c4f9e",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def color(name: str) -> str:
    if name.endswith("(CPU)"):
        return COLORS["CPU"]
    return COLORS.get(name, "#333333")
