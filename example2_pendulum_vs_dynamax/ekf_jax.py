"""The same EKF loss written two ways in plain JAX (no Pallas).

1. ``loss_natural``: the way anyone (and Dynamax) writes it: a state vector, a 2 x 2 covariance matrix,
   ``jax.jacfwd`` for the Jacobians, ``lax.scan`` over time, ``vmap`` over sequences.
2. ``loss_lanes``: the same math written "per row": every scalar is an array over all sequences (a "lane").
   This is the form that our Pallas kernels use; here it runs as ordinary XLA code.

One EKF time step (Dynamax's order: measure, update, predict), with H = dh/dx and F = df/dx:

    S  = H P H^T + r^2                 innovation variance
    e  = y - h(m)                      innovation
    ll = -1/2 (log(2 pi S) + e^2 / S)  log-likelihood of this measurement
    K  = P H^T / S                     Kalman gain
    m  = m + K e,   P = P - K S K^T    update
    m  = f(m),      P = F P F^T + Q    predict
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .pendulum import P0, Q00, Q01, Q11, X0, dynamics, sensor

LOG_2PI = math.log(2.0 * math.pi)        # a Python float: Pallas kernels may not capture JAX array constants


# ------------------------------------------------------------------------------------------ 1. natural form
def log_likelihood_natural(g, r, y):
    """EKF log-likelihood of one sequence y (T,), written with vectors and matrices."""
    Q = jnp.array([[Q00, Q01], [Q01, Q11]])
    f = lambda x: jnp.stack(dynamics(x[0], x[1], g))
    h = lambda x: sensor(x[0])

    def step(carry, y_t):
        m, P = carry
        H = jax.grad(h)(m)                         # (2,) gradient of the scalar sensor
        S = H @ P @ H + r**2
        e = y_t - h(m)
        ll = -0.5 * (LOG_2PI + jnp.log(S) + e**2 / S)
        K = P @ H / S
        m, P = m + K * e, P - S * jnp.outer(K, K)
        F = jax.jacfwd(f)(m)                       # (2, 2) Jacobian of the dynamics
        m, P = f(m), F @ P @ F.T + Q
        return (m, P), ll

    _, ll = jax.lax.scan(step, (jnp.array(X0), P0 * jnp.eye(2)), y)
    return jnp.sum(ll)


def loss_natural(theta, ys):
    """theta = (g, r); ys of shape (T, B)."""
    g, r = theta
    logliks = jax.vmap(log_likelihood_natural, in_axes=(None, None, 1))(g, r, ys)
    return -jnp.mean(logliks) / ys.shape[0]


# ------------------------------------------------------------------------------------------ 2. per-row ("lane") form
def ekf_step_lanes(lanes):
    """One EKF step for many sequences at once.

    lanes in : (m0, m1, p00, p01, p11, y, g, r)  -- mean, covariance (3 unique entries), measurement, parameters
    lanes out: (m0, m1, p00, p01, p11, ll)       -- next predicted mean and covariance, log-likelihood of y
    Each lane is an array of shape (rows,). The Jacobians come from jax.jvp, one direction at a time.
    """
    m0, m1, p00, p01, p11, y, g, r = lanes
    one, zero = jnp.ones_like(m0), jnp.zeros_like(m0)

    # measurement: H = [dh/dangle, 0]
    y_hat, h0 = jax.jvp(sensor, (m0,), (one,))
    S = h0 * p00 * h0 + r * r
    e = y - y_hat
    ll = -0.5 * (LOG_2PI + jnp.log(S) + e * e / S)

    # update
    k0, k1 = p00 * h0 / S, p01 * h0 / S
    m0, m1 = m0 + k0 * e, m1 + k1 * e
    p00, p01, p11 = p00 - k0 * k0 * S, p01 - k0 * k1 * S, p11 - k1 * k1 * S

    # predict: the two columns of F = df/dx, then P = F P F^T + Q
    f = lambda a, b: dynamics(a, b, g)
    (n0, n1), (f00, f10) = jax.jvp(f, (m0, m1), (one, zero))      # column 1 of F
    _, (f01, f11) = jax.jvp(f, (m0, m1), (zero, one))             # column 2 of F
    a00, a01 = f00 * p00 + f01 * p01, f00 * p01 + f01 * p11       # A = F P
    a10, a11 = f10 * p00 + f11 * p01, f10 * p01 + f11 * p11
    p00 = a00 * f00 + a01 * f01 + Q00                              # F P F^T + Q
    p01 = a00 * f10 + a01 * f11 + Q01
    p11 = a10 * f10 + a11 * f11 + Q11
    return n0, n1, p00, p01, p11, ll


def initial_lanes(rows, dtype=jnp.float32):
    """Filter start for ``rows`` sequences: mean X0, covariance P0 * I."""
    full = lambda v: jnp.full((rows,), v, dtype)
    return (full(X0[0]), full(X0[1]), full(P0), full(0.0), full(P0))


def loss_lanes(theta, ys):
    """Same loss as loss_natural, computed with ekf_step_lanes (plain XLA, no Pallas)."""
    g, r = theta
    rows = ys.shape[1]
    gs, rs = jnp.full((rows,), g), jnp.full((rows,), r)

    def step(state, y_t):
        *state, ll = ekf_step_lanes((*state, y_t, gs, rs))
        return tuple(state), ll

    _, ll = jax.lax.scan(step, initial_lanes(rows, ys.dtype), ys)
    return -jnp.mean(jnp.sum(ll, axis=0)) / ys.shape[0]
