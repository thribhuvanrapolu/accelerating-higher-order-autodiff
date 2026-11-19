"""Do all implementations compute the same drone cost and gradient?

Reference: rollout_jax.loss_natural in float64 on the CPU.

    cd jax_pallas && python -m example1_drone_vs_warp.check_correctness
"""

from __future__ import annotations

import json
import pathlib

import common.timing  # noqa: F401  (FP32 precision)
import jax
import jax.numpy as jnp
import numpy as np

from . import rollout_jax, rollout_pallas

JAX_IMPLEMENTATIONS = {
    "JAX natural": rollout_jax.loss_natural,
    "JAX lanes": rollout_jax.loss_lanes,
    "Pallas: whole rollout": rollout_pallas.loss_pallas,
}


def control_points(batch, seed=0):
    """Random control points around hover, like the example's Gaussian sampling (std 0.15, clipped to [0.1, 1])."""
    noise = jax.random.normal(jax.random.PRNGKey(seed), (batch, 4, 4))
    return jnp.clip(0.6 + 0.15 * noise, 0.1, 1.0).astype(jnp.float32)


def compare(name, value, grad, ref_value, ref_grad):
    err_v = abs(value - ref_value) / abs(ref_value)
    err_g = np.max(np.abs(grad - ref_grad)) / np.max(np.abs(ref_grad))
    print(f"{name:24s} cost {value:.6f}  rel.err cost {err_v:.1e}  grad (max-normalised) {err_g:.1e}", flush=True)
    return {"cost": value, "rel_err_cost": err_v, "rel_err_grad": err_g}


def main(batch=16):
    points = control_points(batch)
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        p64 = jnp.asarray(np.asarray(points), jnp.float64)
        ref_value, ref_grad = jax.value_and_grad(rollout_jax.loss_natural)(p64)
        ref_value, ref_grad = float(ref_value), np.asarray(ref_grad)
    print(f"float64 reference cost {ref_value:.8f}")

    results = {"reference_cost": ref_value, "batch": batch}
    for name, loss in JAX_IMPLEMENTATIONS.items():
        value, grad = jax.jit(jax.value_and_grad(loss))(points)
        results[name] = compare(name, float(value), np.asarray(grad), ref_value, ref_grad)

    try:                                                         # the Warp version needs warp-lang
        from .rollout_warp import WarpRollout
        value, grad = WarpRollout(batch).value_and_grad(np.asarray(points))
        results["Warp"] = compare("Warp", float(value), grad, ref_value, ref_grad)
    except ImportError as error:
        print("Warp skipped:", error)

    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "correctness.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
