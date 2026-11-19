"""Part B: uncertainty-aware MPC. The cost now CONTAINS a Jacobian, so training needs a derivative of a derivative.

The drone does not know its start orientation exactly: a small tilt error d_0 (about x, y or z, 0.05 rad).
To first order, an error is carried along the rollout by the linearised dynamics,

    d_{k+1} = F_k d_k,      F_k = d(step)/d(state) at (state_k, u_k)          (a Jacobian-vector product)

which is the same as propagating a covariance P_{k+1} = F_k P_k F_k^T with P_0 = sum_j d_0j d_0j^T. We add the
spread of the position to the cost,

    cost_k += W_UNCERTAINTY * trace(P_k[position]) = W_UNCERTAINTY * sum_j |d_kj[position]|^2,

so the optimiser prefers controls that keep the drone where it should be even if the start tilt is a bit off.
d(cost)/d(controls) now needs d(F_k)/d(controls): second-order automatic differentiation.

* JAX natural: ``jax.jacfwd`` for F_k (13 x 13) and a matrix product, the usual way.
* JAX lanes / Pallas: one row per (rollout, error direction). Each row carries the state and ONE tangent d_kj and
  advances both with a single ``jax.jvp`` of drone.simulate_step; the Pallas backward kernel then runs ``jax.vjp``
  of that (a jvp inside a vjp, inside the kernel). Carrying all 3 tangents in one row made the backward kernel
  ~3x bigger, and Triton needed > 15 minutes to compile it; one tangent per row keeps every kernel small.
* Warp: Warp's released autodiff is first-order reverse mode only, so F_k (or the jvp) would have to be written by
  hand for this step. Not implemented here; see README.md.

    cd jax_pallas && python -m example1_drone_vs_warp.uncertainty      # correctness check -> results/
"""

from __future__ import annotations

import json
import pathlib
import time

import common.timing  # noqa: F401  (FP32 precision)
import jax
import jax.numpy as jnp
import numpy as np

from . import drone as D
from .rollout_jax import START, controls_natural, cost_natural, physics_natural, to_lanes
from .rollout_pallas import rollout_costs

SIGMA = 0.05                 # rad, start tilt uncertainty about each axis
W_UNCERTAINTY = 100.0
DIRECTIONS = 3               # tilt about x, y, z


# ------------------------------------------------------------------------------------------ natural JAX
def start_tangents(dtype):
    """(13, 3): a tilt e_j changes the identity quaternion by 1/2 SIGMA e_j in its vector part."""
    return jnp.zeros((D.STATE_LANES, DIRECTIONS), dtype).at[3, 0].set(0.5 * SIGMA).at[4, 1].set(0.5 * SIGMA).at[5, 2].set(
        0.5 * SIGMA)


def rollout_natural(points):
    def step(carry, k):
        state, tangents = carry
        u = controls_natural(points, k)
        F = jax.jacfwd(physics_natural)(state, u)                    # (13, 13) Jacobian of the step
        state, tangents = physics_natural(state, u), F @ tangents
        spread = jnp.sum(tangents[0:3] ** 2)                         # trace of the position covariance
        return (state, tangents), cost_natural(state, u, k) + W_UNCERTAINTY * spread

    _, costs = jax.lax.scan(step, (START.astype(points.dtype), start_tangents(points.dtype)),
                            jnp.arange(D.HORIZON))
    return jnp.sum(costs)


def loss_natural(points):
    return jnp.sum(jax.vmap(rollout_natural)(points))


# ------------------------------------------------------------------------------------------ per-row form
# Row (b, j) = rollout b followed along error direction j. Its "point" lanes are the 16 control lanes of rollout b,
# plus 1 lane saying whether this row also pays the ordinary cost (only j = 0 does, so it is counted once), plus 3
# lanes with the start tangent of direction j.
ROW_POINT_LANES = D.CONTROL_LANES + 1 + 3


def uncertainty_step(k, lanes, points):
    """One step on lanes: 13 state lanes + 13 tangent lanes. Returns (lanes, cost)."""
    state, tangent = lanes[:D.STATE_LANES], lanes[D.STATE_LANES:]
    controls, base_weight = points[:D.CONTROL_LANES], points[D.CONTROL_LANES]
    u = D.controls_at(controls, k)
    new_state, new_tangent = jax.jvp(lambda s: D.simulate_step(s, u), (state,), (tangent,))   # step and F_k d
    spread = sum(new_tangent[i] * new_tangent[i] for i in range(3))
    cost = base_weight * D.step_cost(new_state, u, k) + W_UNCERTAINTY * spread
    return (*new_state, *new_tangent), cost


def start_lanes(points):
    """Start state at rest, start tangent = the row's tilt direction (in the quaternion's vector part)."""
    state = D.initial_state(points[0].shape[0], points[0].dtype)
    tangent = list(D.initial_state(points[0].shape[0], points[0].dtype))
    tangent[6] = jnp.zeros_like(tangent[6])                          # the start tangent has no q_w part
    tangent[3:6] = points[D.CONTROL_LANES + 1:]
    return (*state, *tangent)


def row_lanes(points):
    """(B, 4, 4) control points -> (20, 3 B) lanes for the rows (b, j), j = 0, 1, 2."""
    batch = points.shape[0]
    controls = jnp.repeat(jnp.stack(to_lanes(points)), DIRECTIONS, axis=1)          # (16, 3B), column 3b + j
    base_weight = jnp.tile(jnp.array([1.0, 0.0, 0.0], points.dtype), batch)[None]   # (1, 3B)
    seeds = jnp.tile(0.5 * SIGMA * jnp.eye(DIRECTIONS, dtype=points.dtype), (1, batch))   # (3, 3B)
    return jnp.concatenate([controls, base_weight, seeds])


def loss_lanes(points):
    lanes = tuple(row_lanes(points))
    _, costs = jax.lax.scan(lambda s, k: uncertainty_step(k, s, lanes), start_lanes(lanes), jnp.arange(D.HORIZON))
    return jnp.sum(costs)


def loss_pallas_unc(points):
    costs = rollout_costs(row_lanes(points), step_fn=uncertainty_step, state_lanes=2 * D.STATE_LANES,
                          point_lanes=ROW_POINT_LANES, start_fn=start_lanes)
    return jnp.sum(costs)


IMPLEMENTATIONS = {
    "JAX natural": loss_natural,
    "JAX lanes": loss_lanes,
    "Pallas: whole rollout": loss_pallas_unc,
}


def main(batch=16):
    from .check_correctness import compare, control_points
    points = control_points(batch)
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        p64 = jnp.asarray(np.asarray(points), jnp.float64)
        ref_value, ref_grad = jax.value_and_grad(loss_natural)(p64)
        ref_value, ref_grad = float(ref_value), np.asarray(ref_grad)
    print(f"float64 reference cost (with uncertainty term) {ref_value:.8f}", flush=True)
    results = {"reference_cost": ref_value, "batch": batch}
    for name, loss in IMPLEMENTATIONS.items():
        tic = time.perf_counter()
        value, grad = jax.jit(jax.value_and_grad(loss))(points)
        print(f"   ({name}: first call incl. compile {time.perf_counter() - tic:.1f} s)", flush=True)
        results[name] = compare(name, float(value), np.asarray(grad), ref_value, ref_grad)
    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "correctness_uncertainty.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
