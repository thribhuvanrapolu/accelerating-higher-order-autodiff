"""The drone MPC loss in plain JAX, two ways.

    loss(points) = sum over rollouts b, steps k of cost(x_k^b, u_k^b),   points of shape (B, 4 control points, 4 props)

1. ``loss_natural``: the way one would normally write it: vectors and quaternions as small jnp arrays,
   ``vmap`` over rollouts, ``lax.scan`` over time.
2. ``loss_lanes``: the per-row form of drone.py (every scalar is an array over rollouts), run by plain XLA.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import drone as D


# ------------------------------------------------------------------------------------------ 1. natural form
def quat_rotate(q, v):
    u, w = q[:3], q[3]
    t = 2.0 * jnp.cross(u, v)
    return v + w * t + jnp.cross(u, t)


def quat_mul(a, b):
    return jnp.concatenate([a[3] * b[:3] + b[3] * a[:3] + jnp.cross(a[:3], b[:3]),
                            jnp.array([a[3] * b[3] - a[:3] @ b[:3]])])


INERTIA = jnp.array(D.INERTIA)
ARM = jnp.array([[0.0, D.ARM, 0.0], [0.0, -D.ARM, 0.0], [D.ARM, 0.0, 0.0], [-D.ARM, 0.0, 0.0]])
SPIN, TARGET, POLE, EZ = jnp.array(D.SPIN), jnp.array(D.TARGET), jnp.array(D.POLE_CENTRE), jnp.array([0.0, 0.0, 1.0])
START = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # x, q, v, w


def controls_natural(points, k):
    """Propeller commands (4,) at step k: linear interpolation of the (4 points, 4 props) control points."""
    weights = jnp.maximum(0.0, 1.0 - jnp.abs(k / D.STEPS_PER_POINT - jnp.arange(D.NUM_POINTS)))
    return D.CONTROL_GAIN * weights @ points


def physics_natural(state, u):
    """One semi-implicit Euler step; state = [x (3), q (4), v (3), w (3)] as one (13,) vector."""
    x, q, v, w = state[0:3], state[3:7], state[7:10], state[10:13]
    force = quat_rotate(q, EZ * D.MAX_THRUST * u.sum())                    # propeller wrench
    torque_b = D.TORQUE_SCALE * (jnp.cross(ARM, EZ * D.MAX_THRUST).T @ u + EZ * D.MAX_TORQUE * (SPIN @ u))
    v = v + (force / D.MASS + EZ * D.GRAVITY) * D.DT
    x = x + v * D.DT
    w_b = quat_rotate(q * jnp.array([-1.0, -1.0, -1.0, 1.0]), w)
    w_b = w_b + (torque_b - jnp.cross(w_b, INERTIA * w_b)) / INERTIA * D.DT
    w = quat_rotate(q, w_b)
    q = q + 0.5 * D.DT * quat_mul(jnp.concatenate([w, jnp.zeros(1)]), q)
    q = q / jnp.linalg.norm(q)
    w = (1.0 - D.ANGULAR_DAMPING * D.DT) * w
    return jnp.concatenate([x, q, v, w])


def cost_natural(state, u, k):
    """Newton's drone_cost + collision_cost after step k."""
    x, q, v, w = state[0:3], state[3:7], state[7:10], state[10:13]
    cost = (D.W_POS * jnp.sum((x - TARGET) ** 2)
            + D.W_ALT * (jnp.maximum(x[2] - 0.75, 0.0) + jnp.maximum(0.25 - x[2], 0.0))
            + D.W_CONTROL * u @ u + D.W_VEL * (v @ v + w @ w) + D.W_UP * (1.0 - quat_rotate(q, EZ)[2]))
    cost = cost * (D.COST_WEIGHT / D.W_TOTAL) * 0.8 ** (D.HORIZON - k - 1.0) / D.HORIZON**2
    local = x - POLE
    closest = jnp.array([0.0, 0.0, jnp.clip(local[2], -D.POLE_HALF_HEIGHT, D.POLE_HALF_HEIGHT)])
    distance = jnp.maximum(jnp.linalg.norm(local - closest) - D.POLE_RADIUS, 0.0)
    return cost + D.COLLISION_WEIGHT * jnp.maximum(D.COLLISION_MARGIN - distance, 0.0)


def rollout_natural(points):
    """Total cost of one rollout; points (4, 4)."""
    def step(state, k):
        u = controls_natural(points, k)
        state = physics_natural(state, u)
        return state, cost_natural(state, u, k)

    _, costs = jax.lax.scan(step, START.astype(points.dtype), jnp.arange(D.HORIZON))
    return jnp.sum(costs)


def loss_natural(points):
    return jnp.sum(jax.vmap(rollout_natural)(points))


# ------------------------------------------------------------------------------------------ 2. per-row form
def to_lanes(points):
    """(B, 4, 4) control points -> tuple of 16 lanes of shape (B,), point-major (lane 4 j + i = point j, prop i)."""
    return tuple(points.reshape(points.shape[0], D.CONTROL_LANES).T)


def loss_lanes(points):
    lanes = to_lanes(points)

    def step(state, k):
        return D.rollout_step(k, state, lanes)

    _, costs = jax.lax.scan(step, D.initial_state(points.shape[0], points.dtype), jnp.arange(D.HORIZON))
    return jnp.sum(costs)
