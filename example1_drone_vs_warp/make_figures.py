"""Figures for Example 1, from the JSON files in results/.   python -m example1_drone_vs_warp.make_figures"""

from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common.plot_style import color, plt
from . import drone as D
from .check_correctness import control_points
from .rollout_jax import START, controls_natural, physics_natural, loss_natural

HERE = pathlib.Path(__file__).parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"


def bar_figure(file, title, out):
    rows = json.loads((RESULTS / file).read_text())
    names = [r["implementation"] for r in rows]
    counts = [r["kernels_per_call"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rows) + 1.0))
    bars = ax.barh(names[::-1], counts[::-1], color=[color(n) for n in names[::-1]])
    ax.set_xscale("log")
    ax.bar_label(bars, labels=[f"{c:,.0f}" for c in counts[::-1]], padding=3)
    ax.set(xlabel="GPU kernels per optimisation step (log scale)", title=f"{title}\nNVIDIA RTX 6000 Ada")
    ax.set_xlim(1, max(counts) * 8)
    fig.savefig(FIGURES / out)


def timing_figure(file, title, out):
    rows = json.loads((RESULTS / file).read_text())
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for name in dict.fromkeys(r["implementation"] for r in rows):
        pts = sorted((r["batch"], r["ms"]) for r in rows if r["implementation"] == name)
        ax.plot(*zip(*pts), "o-", color=color(name), label=name, lw=2, ms=4)
    ax.set(xscale="log", yscale="log", xlabel="number of rollouts B", ylabel="ms per optimisation step", title=f"{title}\nNVIDIA RTX 6000 Ada")
    batches = sorted({r["batch"] for r in rows})
    ax.set_xticks(batches, labels=[f"{b:,}" for b in batches])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.savefig(FIGURES / out)


def trajectory_figure(steps=200):
    """What the optimiser does: one rollout's control points improved by gradient descent (JAX natural code)."""
    points = control_points(1)
    optimizer = optax.adam(0.02)
    state = optimizer.init(points)

    @jax.jit
    def update(points, state):
        grad = jax.grad(loss_natural)(points)
        updates, state = optimizer.update(grad, state)
        return jnp.clip(optax.apply_updates(points, updates), 0.1, 1.0), state

    def path(points):
        def step(s, k):
            s = physics_natural(s, controls_natural(points[0], k))
            return s, s[:3]
        return np.asarray(jax.lax.scan(step, START, jnp.arange(D.HORIZON))[1])

    before = path(points)
    for _ in range(steps):
        points, state = update(points, state)
    after = path(points)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(before[:, 0], before[:, 2], "o-", ms=2, color="#9b9b9b", label="random controls")
    ax.plot(after[:, 0], after[:, 2], "o-", ms=2, color="#1f5fa8", label=f"after {steps} gradient steps")
    ax.plot(*D.TARGET[::2], "*", ms=14, color="#c0392b", label="target")
    ax.set(xlabel="x (m)", ylabel="height z (m)", title="One rollout, 30 steps (0.5 s)")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "trajectory.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    trajectory_figure()
    jobs = (
        (bar_figure, "kernels.json", "Part A: kernels per optimisation step (B = 16, 30 steps)", "kernels.png"),
        (timing_figure, "timings.json", "Part A: drone MPC, first-order gradient", "time_vs_batch.png"),
        (bar_figure, "kernels_uncertainty.json", "Part B: kernels per step, cost with a Jacobian inside",
         "kernels_uncertainty.png"),
        (timing_figure, "timings_uncertainty.json", "Part B: second-order gradient (no Warp version)",
         "time_vs_batch_uncertainty.png"),
    )
    for make, file, title, out in jobs:
        if (RESULTS / file).exists():
            make(file, title, out)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
