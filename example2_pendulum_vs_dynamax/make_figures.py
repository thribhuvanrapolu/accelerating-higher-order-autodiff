"""Figures for Example 2, from the JSON files in results/.   python -m example2_pendulum_vs_dynamax.make_figures"""

from __future__ import annotations

import json
import pathlib

import jax
import numpy as np

from common.plot_style import color, plt
from .pendulum import DT, G_TRUE, R_TRUE, simulate

HERE = pathlib.Path(__file__).parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"


def data_figure():
    ys, states = simulate(jax.random.PRNGKey(0), 1, return_states=True)
    t = np.arange(ys.shape[0]) * DT
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(t, np.sin(states[:, 0, 0]), color="#1f5fa8", lw=2, label=r"true $\sin\alpha$")
    ax.plot(t, ys[:, 0], "o", ms=2, color="#9b9b9b", label=r"measurement $y_k$")
    ax.set(xlabel="time (s)", ylabel="sensor value", title="The data: a noisy pendulum sensor (T = 400 steps)")
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(FIGURES / "pendulum_data.png")


def kernels_figure():
    rows = json.loads((RESULTS / "kernels.json").read_text())
    names = [r["implementation"] for r in rows]
    counts = [r["kernels_per_call"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    bars = ax.barh(names[::-1], counts[::-1], color=[color(n) for n in names[::-1]])
    ax.set_xscale("log")
    ax.bar_label(bars, labels=[f"{c:,.0f}" for c in counts[::-1]], padding=3)
    ax.set(xlabel="GPU kernels per training step (log scale)",
           title=f"Kernels launched by one loss + gradient (B = {rows[0]['batch']}, T = 400)")
    ax.set_xlim(1, max(counts) * 8)
    fig.savefig(FIGURES / "kernels.png")


def timing_figure():
    rows = json.loads((RESULTS / "timings.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name in dict.fromkeys(r["implementation"] for r in rows):
        for device in ("gpu", "cpu"):
            pts = sorted((r["batch"], r["ms"]) for r in rows if r["implementation"] == name and r["device"] == device)
            if pts:
                label = name if device == "gpu" else f"{name} (CPU)"
                ax.plot(*zip(*pts), "o-" if device == "gpu" else "s--", color=color(label) if device == "cpu"
                        else color(name), label=label, lw=2, ms=4, alpha=0.6 if name == "JAX natural" and
                        device == "cpu" else 1.0)
    ax.set(xscale="log", yscale="log", xlabel="number of sequences B", ylabel="ms per training step",
           title="Time of one loss + gradient (A100 unless marked CPU)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES / "time_vs_batch.png")


def fit_figure():
    runs = json.loads((RESULTS / "fit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for run in runs:
        h = np.array(run["history"])
        label = f"{run['implementation']} ({run['device'].upper()})"
        c = color(run["implementation"]) if run["device"] == "gpu" else color(label)
        axes[0].plot(h[:, 3], h[:, 0], color=c, lw=2, label=f"{label}: {h[-1, 3]:.1f} s")
        axes[1].plot(np.arange(1, len(h) + 1), h[:, 0], color=c, lw=2)
    axes[0].axhline(G_TRUE, color="k", lw=0.8, ls=":")
    axes[1].axhline(G_TRUE, color="k", lw=0.8, ls=":")
    axes[0].set(xscale="log", xlabel="wall-clock time (s)", ylabel="learned g", title="Learning g: time")
    axes[1].set(xlabel="Adam step", title="Learning g: same steps, same result")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "fit_gravity.png")
    print("final (g, r):", {r["implementation"]: tuple(np.round(r["history"][-1][:2], 4)) for r in runs},
          f"true ({G_TRUE}, {R_TRUE})")


def main():
    FIGURES.mkdir(exist_ok=True)
    data_figure()
    for make, needs in ((kernels_figure, "kernels.json"), (timing_figure, "timings.json"), (fit_figure, "fit.json")):
        if (RESULTS / needs).exists():
            make()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
