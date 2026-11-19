"""The EKF loss with Pallas kernels, in two levels.

Level 1  ``loss_pallas_step``: one fused kernel per EKF time step (forward) and one per step (backward),
         still driven by ``lax.scan``.            T time steps -> about 2 T kernels (+ a little glue).
Level 2  ``loss_pallas_loop``: the WHOLE filter, all T time steps, in ONE forward kernel and ONE backward kernel.
                                                  T time steps -> 2 kernels.

Both reuse ``ekf_step_lanes`` from ekf_jax.py unchanged, and both get their derivatives from ``jax.vjp`` run inside
the kernel: no derivative is written by hand.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from common.row_kernel import make_row_op
from .ekf_jax import ekf_step_lanes, initial_lanes

STATE_LANES = 5          # m0, m1, p00, p01, p11


def pad_rows(ys, block):
    """Pad the number of sequences B up to a multiple of ``block`` (padded sequences are ignored in the loss)."""
    batch = ys.shape[1]
    padded = -(-batch // block) * block
    return jnp.pad(ys, ((0, 0), (0, padded - batch))), batch


def pick_block(batch):
    """Sequences handled by one kernel program: a power of two, at most 32."""
    return min(32, 1 << max(0, (batch - 1).bit_length()))


# ------------------------------------------------------------------------------------------ level 1: kernel per step
def loss_pallas_step(theta, ys):
    g, r = theta
    block = pick_block(ys.shape[1])
    ys, batch = pad_rows(ys, block)
    rows = ys.shape[1]
    step_op = make_row_op(ekf_step_lanes, n_in=8, n_out=6, block=block)
    gs, rs = jnp.full((rows,), g), jnp.full((rows,), r)

    def step(state, y_t):
        out = step_op(jnp.stack([*state, y_t, gs, rs]))       # (6, rows): new state + log-likelihood
        return tuple(out[:STATE_LANES]), out[STATE_LANES]

    _, ll = jax.lax.scan(step, initial_lanes(rows, ys.dtype), ys)
    return -jnp.sum(ll[:, :batch]) / (batch * ys.shape[0])


# ------------------------------------------------------------------------------------------ level 2: whole loop
def make_filter_op(num_steps: int, block: int):
    """op(ys (T, N), g (1, N), r (1, N)) -> total log-likelihood per sequence (1, N), in one kernel each way."""

    def forward_kernel(ys_ref, g_ref, r_ref, ll_ref, saved_ref):
        g, r = g_ref[0, :], r_ref[0, :]

        def body(t, carry):
            state, ll_sum = carry
            for k in range(STATE_LANES):                      # keep the state at the start of step t for backward
                saved_ref[t, k, :] = state[k]
            *state, ll = ekf_step_lanes((*state, ys_ref[t, :], g, r))
            return tuple(state), ll_sum + ll

        start = initial_lanes(block, g.dtype)
        _, ll_sum = jax.lax.fori_loop(0, num_steps, body, (start, jnp.zeros_like(g)))
        ll_ref[0, :] = ll_sum

    def backward_kernel(ys_ref, g_ref, r_ref, saved_ref, ll_bar_ref, g_bar_ref, r_bar_ref):
        g, r, ll_bar = g_ref[0, :], r_ref[0, :], ll_bar_ref[0, :]

        def body(i, carry):                                   # walk the time steps backwards
            t = num_steps - 1 - i
            state_bar, g_bar, r_bar = carry
            lanes = (*(saved_ref[t, k, :] for k in range(STATE_LANES)), ys_ref[t, :], g, r)
            _, pullback = jax.vjp(ekf_step_lanes, lanes)      # autodiff of ONE step, inside the kernel
            (lanes_bar,) = pullback((*state_bar, ll_bar))
            return tuple(lanes_bar[:STATE_LANES]), g_bar + lanes_bar[6], r_bar + lanes_bar[7]

        zero = jnp.zeros_like(g)
        start = (tuple(zero for _ in range(STATE_LANES)), zero, zero)   # the last state does not enter the loss
        _, g_bar, r_bar = jax.lax.fori_loop(0, num_steps, body, start)
        g_bar_ref[0, :] = g_bar
        r_bar_ref[0, :] = r_bar

    # block specs: program i sees columns [i * block, (i + 1) * block) of every array
    row = pl.BlockSpec((1, block), lambda i: (0, i))
    series = pl.BlockSpec((num_steps, block), lambda i: (0, i))
    saved = pl.BlockSpec((num_steps, STATE_LANES, block), lambda i: (0, 0, i))

    def run_forward(ys, g, r):
        rows = ys.shape[1]
        return pl.pallas_call(
            forward_kernel, grid=(rows // block,),
            in_specs=[series, row, row], out_specs=[row, saved],
            out_shape=[jax.ShapeDtypeStruct((1, rows), ys.dtype),
                       jax.ShapeDtypeStruct((num_steps, STATE_LANES, rows), ys.dtype)],
        )(ys, g, r)

    def run_backward(ys, g, r, saved_states, ll_bar):
        rows = ys.shape[1]
        return pl.pallas_call(
            backward_kernel, grid=(rows // block,),
            in_specs=[series, row, row, saved, row], out_specs=[row, row],
            out_shape=[jax.ShapeDtypeStruct((1, rows), ys.dtype)] * 2,
        )(ys, g, r, saved_states, ll_bar)

    @jax.custom_vjp
    def op(ys, g, r):
        return run_forward(ys, g, r)[0]

    def op_fwd(ys, g, r):
        ll, saved_states = run_forward(ys, g, r)
        return ll, (ys, g, r, saved_states)

    def op_bwd(residuals, ll_bar):
        ys, g, r, saved_states = residuals
        g_bar, r_bar = run_backward(ys, g, r, saved_states, ll_bar)
        return jnp.zeros_like(ys), g_bar, r_bar               # the data ys needs no gradient

    op.defvjp(op_fwd, op_bwd)
    return op


def loss_pallas_loop(theta, ys):
    g, r = theta
    block = pick_block(ys.shape[1])
    ys, batch = pad_rows(ys, block)
    rows = ys.shape[1]
    filter_op = make_filter_op(ys.shape[0], block)
    ll = filter_op(ys, jnp.full((1, rows), g), jnp.full((1, rows), r))
    return -jnp.sum(ll[0, :batch]) / (batch * ys.shape[0])
