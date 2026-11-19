"""Do all implementations compute the same loss and gradient?

Reference: our natural EKF in float64 on the CPU. Every GPU version runs in float32 and should agree with it to
float32 round-off (relative error ~1e-6 .. 1e-5).

    cd jax_pallas && python -m example2_pendulum_vs_dynamax.check_correctness
"""

from __future__ import annotations

import json
import pathlib

import common.timing  # noqa: F401  (sets FP32 matmul precision)
import jax
import jax.numpy as jnp
import numpy as np

from . import ekf_dynamax, ekf_jax, ekf_pallas
from .pendulum import simulate

IMPLEMENTATIONS = {
    "Dynamax": ekf_dynamax.loss,
    "JAX natural": ekf_jax.loss_natural,
    "JAX lanes": ekf_jax.loss_lanes,
    "Pallas: kernel per step": ekf_pallas.loss_pallas_step,
    "Pallas: whole loop": ekf_pallas.loss_pallas_loop,
}


def main(batch=64):
    theta = (8.0, 0.4)                                        # a point away from the optimum (g = 9.8, r = 0.3)
    ys = simulate(jax.random.PRNGKey(0), batch)

    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        ys64 = jnp.asarray(np.asarray(ys), jnp.float64)
        ref_loss, ref_grad = jax.value_and_grad(ekf_jax.loss_natural)((jnp.float64(theta[0]), jnp.float64(theta[1])), ys64)
        ref_grad = np.array([float(v) for v in ref_grad])
    print(f"float64 reference: loss {float(ref_loss):.10f}  grad {ref_grad}")

    results = {"reference_loss": float(ref_loss), "reference_grad": ref_grad.tolist(), "batch": batch}
    for name, loss in IMPLEMENTATIONS.items():
        value, grad = jax.jit(jax.value_and_grad(loss))((jnp.float32(theta[0]), jnp.float32(theta[1])), ys)
        grad = np.array([float(v) for v in grad])
        err_loss = abs(float(value) - float(ref_loss)) / abs(float(ref_loss))
        err_grad = np.max(np.abs(grad - ref_grad) / np.abs(ref_grad))
        results[name] = {"loss": float(value), "grad": grad.tolist(), "rel_err_loss": err_loss, "rel_err_grad": err_grad}
        print(f"{name:26s} loss {float(value):.8f}  grad {grad}  rel.err loss {err_loss:.1e}  grad {err_grad:.1e}")

    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "correctness.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
