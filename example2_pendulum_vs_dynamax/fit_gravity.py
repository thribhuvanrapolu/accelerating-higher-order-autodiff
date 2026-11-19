"""Learn gravity g and sensor noise r from noisy pendulum measurements, with each implementation.

Every implementation starts at (g, r) = (8.0, 0.4), takes the same Adam steps on the same data, and should land on
(about) the true (9.8, 0.3). We record the parameters after every step and the wall-clock time.

    cd jax_pallas && python -m example2_pendulum_vs_dynamax.fit_gravity     -> results/fit.json
"""

from __future__ import annotations

import json
import pathlib
import time

import common.timing  # noqa: F401  (FP32 precision)
import jax
import jax.numpy as jnp
import optax

from .check_correctness import IMPLEMENTATIONS
from .pendulum import simulate

STEPS, BATCH, LEARNING_RATE = 300, 16, 0.05
RUNS = (("Dynamax", "gpu"), ("JAX natural", "cpu"), ("Pallas: whole loop", "gpu"))


def fit(name, device):
    with jax.default_device(jax.devices(device)[0]):
        ys = simulate(jax.random.PRNGKey(0), BATCH)
        theta = (jnp.float32(8.0), jnp.float32(0.4))
        optimizer = optax.adam(LEARNING_RATE)
        state = optimizer.init(theta)

        @jax.jit
        def train_step(theta, state):
            loss, grad = jax.value_and_grad(IMPLEMENTATIONS[name])(theta, ys)
            updates, state = optimizer.update(grad, state)
            return optax.apply_updates(theta, updates), state, loss

        jax.block_until_ready(train_step(theta, state))          # compile outside the clock
        history, tic = [], time.perf_counter()
        for _ in range(STEPS):
            theta, state, loss = train_step(theta, state)
            history.append((float(theta[0]), float(theta[1]), float(loss), time.perf_counter() - tic))
    print(f"{name:20s} {device}: g = {history[-1][0]:.4f}, r = {history[-1][1]:.4f} in {history[-1][3]:.2f} s")
    return {"implementation": name, "device": device, "history": history}


def main():
    results = [fit(name, device) for name, device in RUNS]
    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "fit.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
