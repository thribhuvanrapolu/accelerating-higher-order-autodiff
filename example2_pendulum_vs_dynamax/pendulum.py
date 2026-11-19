"""The pendulum of Dynamax's EKF notebook (Saerkkae, Bayesian Filtering and Smoothing, Example 3.7).

    state      x = [angle, angular velocity]
    dynamics   x_{k+1} = f(x_k) + noise,   f(x) = [x0 + x1 dt,  x1 - g sin(x0) dt]
    sensor     y_k     = h(x_k) + noise,   h(x) = sin(x0)

We learn the gravity g and the sensor noise r from noisy measurements y_{1:T}.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# --- the numbers from Dynamax's notebook docs/notebooks/nonlinear_gaussian_ssm/ekf_ukf_pendulum.ipynb
DT = 0.0125              # time step (s)
G_TRUE = 9.8             # gravity (m/s^2)        -> learned
Q_C = 1.0                # process noise intensity
R_TRUE = 0.3             # sensor noise std        -> learned
NUM_STEPS = 400          # 5 seconds of data
X0 = (jnp.pi / 2, 0.0)   # initial state (also the filter's initial mean)
P0 = 0.1                 # filter's initial covariance = 0.1 * I

# process-noise covariance of the discretised pendulum (2 x 2, symmetric)
Q00, Q01, Q11 = Q_C * DT**3 / 3, Q_C * DT**2 / 2, Q_C * DT


def dynamics(angle, rate, g):
    """f(x): one Euler step of the pendulum. Works on scalars or on arrays of many pendulums at once."""
    return angle + rate * DT, rate - g * jnp.sin(angle) * DT


def sensor(angle):
    """h(x): we only measure sin(angle)."""
    return jnp.sin(angle)


def simulate(key, num_sequences: int, num_steps: int = NUM_STEPS, g: float = G_TRUE, r: float = R_TRUE,
             return_states: bool = False):
    """Noisy measurements y of shape (num_steps, num_sequences), like the notebook's simulate_pendulum.
    With ``return_states`` also the true states (num_steps, 2, num_sequences)."""
    q_chol = jnp.linalg.cholesky(jnp.array([[Q00, Q01], [Q01, Q11]]))

    def step(x, key):
        k1, k2 = jax.random.split(key)
        angle, rate = dynamics(x[0], x[1], g)
        noise = q_chol @ jax.random.normal(k1, (2, num_sequences))
        x = jnp.stack([angle, rate]) + noise
        y = sensor(x[0]) + r * jax.random.normal(k2, (num_sequences,))
        return x, (y, x)

    x_start = jnp.tile(jnp.array(X0)[:, None], (1, num_sequences))
    _, (ys, states) = jax.lax.scan(step, x_start, jax.random.split(key, num_steps))
    return (ys, states) if return_states else ys
