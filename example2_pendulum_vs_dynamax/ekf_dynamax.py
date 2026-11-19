"""Baseline: the loss computed with Dynamax's own extended Kalman filter, called exactly as its notebook does.

loss(g, r) = - (1 / (B T)) * sum over sequences of log p(y_{1:T} | g, r)     (the EKF marginal log-likelihood)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter

from .pendulum import DT, P0, Q00, Q01, Q11, X0


def log_likelihood(g, r, y):
    """Dynamax EKF log-likelihood of one measurement sequence y (T,)."""
    params = ParamsNLGSSM(
        initial_mean=jnp.array(X0),
        initial_covariance=P0 * jnp.eye(2),
        dynamics_function=lambda x: jnp.array([x[0] + x[1] * DT, x[1] - g * jnp.sin(x[0]) * DT]),
        dynamics_covariance=jnp.array([[Q00, Q01], [Q01, Q11]]),
        emission_function=lambda x: jnp.sin(x[:1]),
        emission_covariance=(r**2) * jnp.eye(1),
    )
    return extended_kalman_filter(params, y[:, None]).marginal_loglik


def loss(theta, ys):
    """theta = (g, r); ys of shape (T, B)."""
    g, r = theta
    logliks = jax.vmap(log_likelihood, in_axes=(None, None, 1))(g, r, ys)
    return -jnp.mean(logliks) / ys.shape[0]
