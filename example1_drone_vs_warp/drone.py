"""The drone of NVIDIA Newton's official example ``newton/examples/diffsim/example_diffsim_drone.py``.

A quadrotor (one rigid body, 4 propellers) must fly to a target. Its 4 propeller commands are given by 4 control
points per propeller, linearly interpolated over a 30-step horizon. Model-predictive control improves the control
points by gradient descent THROUGH the simulation: we need d(total cost)/d(control points).

Everything here is written "per row": every scalar is an array over many rollouts (a "lane"), so the same code runs
as plain JAX (XLA) and inside a Pallas kernel. The numbers are the ones of the Newton example (Newton 1.8).

State lanes (13):  position x (3), orientation quaternion q = (qx, qy, qz, qw) (4), velocity v (3), angular velocity w
                   (3, world frame).
Controls:          4 control points x 4 propellers = 16 lanes, interpolated to u_1..u_4 at every step.

Attribution: the physics, constants and cost are ported from Newton's example_diffsim_drone.py,
Copyright (c) 2025 The Newton Developers, licensed under the Apache License 2.0
(https://github.com/newton-physics/newton). Rewritten here in JAX "per row" form.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

# --------------------------------------------------------------------------------------- constants (Newton example)
FPS = 60.0
DT = 1.0 / FPS                    # simulation step (s)
HORIZON = 30                      # simulation steps per rollout (3 control intervals x 10 steps)
STEPS_PER_POINT = 10              # simulation steps between control points
NUM_POINTS = 4                    # control points per propeller (3 intervals + 1)
CONTROL_GAIN = 0.8                # propeller command = gain x interpolated control point
MASS = 0.56                       # kg   (two carbon-fibre cross bars, density 1750)
INERTIA = (0.00376133, 0.00376133, 0.00748533)   # kg m^2, diagonal
GRAVITY = -9.81                   # m/s^2 along z
ANGULAR_DAMPING = 0.05            # Newton's SolverSemiImplicit default
MAX_THRUST = 4.179446             # N per propeller at command 1
MAX_TORQUE = 0.055562183          # N m (propeller drag torque) at command 1
ARM = 0.2                         # m, propeller distance from the centre
TORQUE_SCALE = 0.8                # Newton applies "torque *= 0.8" to every propeller wrench
SPIN = (-1.0, 1.0, 1.0, -1.0)     # turning direction of propellers at (0,+L), (0,-L), (+L,0), (-L,0)

TARGET = (1.0, 0.0, 0.5)          # first target of the example
POLE_CENTRE, POLE_RADIUS, POLE_HALF_HEIGHT = (0.5, 0.5, 2.0), 0.15, 2.0      # capsule obstacle along z
COLLISION_MARGIN, COLLISION_WEIGHT, COST_WEIGHT = 0.2, 1e4, 1e3
W_POS, W_ALT, W_CONTROL, W_VEL, W_UP = 1000.0, 100.0, 0.05, 0.1, 10.0
W_TOTAL = W_POS + W_ALT + W_CONTROL + W_VEL + W_UP

LOG_08 = math.log(0.8)
STATE_LANES = 13
CONTROL_LANES = NUM_POINTS * 4


# --------------------------------------------------------------------------------------- small 3-D helpers on lanes
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rotate(q, v):
    """Rotate vector v by unit quaternion q = (x, y, z, w):  v + 2 w (u x v) + 2 u x (u x v), u = (x, y, z)."""
    u, w = q[:3], q[3]
    t = tuple(2.0 * c for c in cross(u, v))
    s = cross(u, t)
    return tuple(v[k] + w * t[k] + s[k] for k in range(3))


def rotate_inverse(q, v):
    return rotate((-q[0], -q[1], -q[2], q[3]), v)


def quat_times(a, b):
    """Hamilton product a * b of quaternions stored as (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


# --------------------------------------------------------------------------------------- controls
def controls_at(points, step):
    """Propeller commands u_1..u_4 at simulation step ``step`` (may be a traced integer).

    Linear interpolation between control points, written with "hat" weights so there is no branching:
    weight of point j at time t is max(0, 1 - |t - j|), with t = step / STEPS_PER_POINT.
    ``points`` holds 16 lanes, ordered point-major: points[4 j + i] = point j of propeller i.
    """
    t = step / STEPS_PER_POINT
    weights = [jnp.maximum(0.0, 1.0 - jnp.abs(t - j)) for j in range(NUM_POINTS)]
    return tuple(CONTROL_GAIN * sum(weights[j] * points[4 * j + i] for j in range(NUM_POINTS)) for i in range(4))


# --------------------------------------------------------------------------------------- one simulation step
def simulate_step(state, u):
    """Semi-implicit Euler step of Newton's SolverSemiImplicit (rigid body, centre of mass at the origin).

        v+ = v + (F / m + g) dt                        x+ = x + v+ dt
        w_b = R^T w,  tau_b = torque_b - w_b x (I w_b)  w+ = R (w_b + I^-1 tau_b dt)
        q+ = normalise(q + 1/2 (w+, 0) * q dt)         w+ <- (1 - damping dt) w+
    """
    x, q, v, w = state[0:3], state[3:7], state[7:10], state[10:13]

    # propeller wrench: thrust along the body z axis; torque in the body frame
    thrust = MAX_THRUST * (u[0] + u[1] + u[2] + u[3])
    force = rotate(q, (0.0, 0.0, thrust))
    torque_b = (TORQUE_SCALE * MAX_THRUST * ARM * (u[0] - u[1]),
                TORQUE_SCALE * MAX_THRUST * ARM * (u[3] - u[2]),
                TORQUE_SCALE * MAX_TORQUE * sum(SPIN[i] * u[i] for i in range(4)))

    # linear part
    v_new = (v[0] + force[0] / MASS * DT, v[1] + force[1] / MASS * DT, v[2] + (force[2] / MASS + GRAVITY) * DT)
    x_new = tuple(x[k] + v_new[k] * DT for k in range(3))

    # angular part, in the body frame
    w_b = rotate_inverse(q, w)
    gyro = cross(w_b, tuple(INERTIA[k] * w_b[k] for k in range(3)))
    w_b = tuple(w_b[k] + (torque_b[k] - gyro[k]) / INERTIA[k] * DT for k in range(3))
    w_new = rotate(q, w_b)

    # orientation update, then normalise
    dq = quat_times((*w_new, 0.0), q)                  # (w, 0) * q
    q_new = tuple(q[k] + 0.5 * DT * dq[k] for k in range(4))
    norm = jnp.sqrt(sum(c * c for c in q_new))
    q_new = tuple(c / norm for c in q_new)

    w_new = tuple((1.0 - ANGULAR_DAMPING * DT) * c for c in w_new)
    return (*x_new, *q_new, *v_new, *w_new)


# --------------------------------------------------------------------------------------- the cost of one step
def step_cost(state, u, step):
    """Newton's drone_cost + collision_cost after simulation step ``step``."""
    x, q, v, w = state[0:3], state[3:7], state[7:10], state[10:13]
    position = sum((x[k] - TARGET[k]) ** 2 for k in range(3))
    altitude = jnp.maximum(x[2] - 0.75, 0.0) + jnp.maximum(0.25 - x[2], 0.0)
    upright = 1.0 - rotate(q, (0.0, 0.0, 1.0))[2]       # 1 - (body z axis) . (world z axis)
    velocity = sum(c * c for c in v) + sum(c * c for c in w)
    control = sum(c * c for c in u)
    discount = jnp.exp(LOG_08 * (HORIZON - step - 1.0)) / HORIZON**2           # 0.8^(H - step - 1) / H^2
    cost = (W_POS * position + W_ALT * altitude + W_CONTROL * control + W_VEL * velocity + W_UP * upright)
    cost = cost * (COST_WEIGHT / W_TOTAL) * discount

    # obstacle: distance from the drone centre to a vertical capsule; penalised inside the margin
    local = tuple(x[k] - POLE_CENTRE[k] for k in range(3))
    along = jnp.clip(local[2], -POLE_HALF_HEIGHT, POLE_HALF_HEIGHT)
    distance = jnp.sqrt(local[0] ** 2 + local[1] ** 2 + (local[2] - along) ** 2) - POLE_RADIUS
    distance = jnp.maximum(distance, 0.0)
    return cost + COLLISION_WEIGHT * jnp.maximum(COLLISION_MARGIN - distance, 0.0)


def rollout_step(step, state, points):
    """Controls -> simulate -> cost, for simulation step ``step``. Returns (next state, cost of this step)."""
    u = controls_at(points, step)
    state = simulate_step(state, u)
    return state, step_cost(state, u, step)


def initial_state(rows, dtype=jnp.float32):
    """All rollouts start at rest at the origin, level (quaternion (0, 0, 0, 1))."""
    zero, one = jnp.zeros((rows,), dtype), jnp.ones((rows,), dtype)
    return (zero, zero, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, zero)
