"""The drone MPC loss as ONE forward and ONE backward Pallas kernel for the whole 30-step rollout.

Forward kernel : for k = 0..29: save state_k; (state_{k+1}, cost_k) = rollout_step(k, state_k, points)
Backward kernel: for k = 29..0: load state_k; run jax.vjp of rollout_step INSIDE the kernel to push the cost
                 gradient back through step k, and add up d(cost)/d(points).
The physics is drone.py, unchanged; no derivative is written by hand.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import drone as D
from .rollout_jax import to_lanes


def drone_start(points):
    """Start state of drone.py for a block of rows (rows and dtype taken from the control lanes)."""
    return D.initial_state(points[0].shape[0], points[0].dtype)


def make_rollout_op(block: int, step_fn=D.rollout_step, state_lanes: int = D.STATE_LANES,
                    point_lanes: int = D.CONTROL_LANES, start_fn=drone_start):
    """op(points (point_lanes, N)) -> total cost per row (1, N).

    ``step_fn(k, state, points) -> (state, cost)`` is one step on lanes; ``start_fn(points)`` gives the start state.
    The defaults are the drone of drone.py; uncertainty.py passes a step that also carries a tangent vector, and
    extra "point" lanes that tell each row which tangent direction it follows.
    """
    H, S, P = D.HORIZON, state_lanes, point_lanes

    def forward_kernel(points_ref, cost_ref, saved_ref):
        points = tuple(points_ref[k, :] for k in range(P))

        def body(k, carry):
            state, total = carry
            for i in range(S):                                   # keep the state at the start of step k
                saved_ref[k, i, :] = state[i]
            state, cost = step_fn(k, state, points)
            return state, total + cost

        start = start_fn(points)
        _, total = jax.lax.fori_loop(0, H, body, (start, jnp.zeros_like(points[0])))
        cost_ref[0, :] = total

    def backward_kernel(points_ref, saved_ref, cost_bar_ref, points_bar_ref):
        points = tuple(points_ref[k, :] for k in range(P))
        cost_bar = cost_bar_ref[0, :]

        def body(i, carry):                                      # walk the steps backwards
            k = H - 1 - i
            state_bar, points_bar = carry
            state = tuple(saved_ref[k, j, :] for j in range(S))
            _, pullback = jax.vjp(lambda s, p: step_fn(k, s, p), state, points)      # autodiff in the kernel
            state_bar, step_points_bar = pullback((state_bar, cost_bar))
            return state_bar, tuple(a + b for a, b in zip(points_bar, step_points_bar))

        zero = jnp.zeros_like(cost_bar)
        start = (tuple(zero for _ in range(S)), tuple(zero for _ in range(P)))
        _, points_bar = jax.lax.fori_loop(0, H, body, start)
        for k in range(P):
            points_bar_ref[k, :] = points_bar[k]

    lanes = lambda height: pl.BlockSpec((height, block), lambda i: (0, i))
    saved = pl.BlockSpec((H, S, block), lambda i: (0, 0, i))

    def run_forward(points):
        rows = points.shape[1]
        return pl.pallas_call(
            forward_kernel, grid=(rows // block,), in_specs=[lanes(P)], out_specs=[lanes(1), saved],
            out_shape=[jax.ShapeDtypeStruct((1, rows), points.dtype),
                       jax.ShapeDtypeStruct((H, S, rows), points.dtype)],
        )(points)

    def run_backward(points, saved_states, cost_bar):
        rows = points.shape[1]
        return pl.pallas_call(
            backward_kernel, grid=(rows // block,), in_specs=[lanes(P), saved, lanes(1)],
            out_specs=lanes(P), out_shape=jax.ShapeDtypeStruct(points.shape, points.dtype),
        )(points, saved_states, cost_bar)

    @jax.custom_vjp
    def op(points):
        return run_forward(points)[0]

    def op_fwd(points):
        cost, saved_states = run_forward(points)
        return cost, (points, saved_states)

    def op_bwd(residuals, cost_bar):
        return (run_backward(*residuals, cost_bar),)

    op.defvjp(op_fwd, op_bwd)
    return op


def rollout_costs(lanes, **rollout):
    """Total cost of every column of ``lanes`` (point_lanes, N). N is padded to a multiple of the block size."""
    rows = lanes.shape[1]
    block = min(32, 1 << max(0, (rows - 1).bit_length()))
    padded = -(-rows // block) * block
    lanes = jnp.pad(lanes, ((0, 0), (0, padded - rows)), mode="edge")     # padded columns are ignored
    return make_rollout_op(block, **rollout)(lanes)[0, :rows]


def loss_pallas(points):
    """Same loss as rollout_jax.loss_natural; points (B, 4, 4)."""
    return jnp.sum(rollout_costs(jnp.stack(to_lanes(points))))
