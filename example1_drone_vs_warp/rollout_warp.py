"""The drone MPC loss in NVIDIA Warp, written the way Warp/Newton examples do it.

* one Warp kernel per simulation step; thread b simulates rollout b (same physics and cost as drone.py)
* ``wp.Tape`` records the launches; ``tape.backward`` runs Warp's automatically generated adjoint kernels
* the whole forward + backward pass is captured once as a CUDA graph and replayed (``wp.capture_launch``)

Every step writes its state into a new slot (index k + 1), because Warp's adjoints need the inputs of every step and
must not see them overwritten.

Attribution: physics and cost ported from Newton's example_diffsim_drone.py, Copyright (c) 2025 The Newton
Developers, Apache License 2.0 (https://github.com/newton-physics/newton).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from . import drone as D

wp.config.quiet = True

# Warp kernels read plain float constants (not Python tuples)
IXX, IYY, IZZ = (wp.constant(c) for c in D.INERTIA)
TX, TY, TZ = (wp.constant(c) for c in D.TARGET)
PX, PY, PZ = (wp.constant(c) for c in D.POLE_CENTRE)


@wp.kernel
def drone_step(points: wp.array3d(dtype=float), k: int,
               x: wp.array2d(dtype=wp.vec3), q: wp.array2d(dtype=wp.quat),
               v: wp.array2d(dtype=wp.vec3), w: wp.array2d(dtype=wp.vec3),
               cost: wp.array(dtype=float)):
    b = wp.tid()

    # --- controls: linear interpolation of the 4 control points ("hat" weights), times the gain
    t = float(k) / float(D.STEPS_PER_POINT)
    a0 = wp.max(0.0, 1.0 - wp.abs(t - 0.0))
    a1 = wp.max(0.0, 1.0 - wp.abs(t - 1.0))
    a2 = wp.max(0.0, 1.0 - wp.abs(t - 2.0))
    a3 = wp.max(0.0, 1.0 - wp.abs(t - 3.0))
    u0 = D.CONTROL_GAIN * (a0 * points[b, 0, 0] + a1 * points[b, 1, 0] + a2 * points[b, 2, 0] + a3 * points[b, 3, 0])
    u1 = D.CONTROL_GAIN * (a0 * points[b, 0, 1] + a1 * points[b, 1, 1] + a2 * points[b, 2, 1] + a3 * points[b, 3, 1])
    u2 = D.CONTROL_GAIN * (a0 * points[b, 0, 2] + a1 * points[b, 1, 2] + a2 * points[b, 2, 2] + a3 * points[b, 3, 2])
    u3 = D.CONTROL_GAIN * (a0 * points[b, 0, 3] + a1 * points[b, 1, 3] + a2 * points[b, 2, 3] + a3 * points[b, 3, 3])

    # --- propeller wrench
    r = q[k, b]
    force = wp.quat_rotate(r, wp.vec3(0.0, 0.0, D.MAX_THRUST * (u0 + u1 + u2 + u3)))
    torque_b = wp.vec3(D.TORQUE_SCALE * D.MAX_THRUST * D.ARM * (u0 - u1),
                       D.TORQUE_SCALE * D.MAX_THRUST * D.ARM * (u3 - u2),
                       D.TORQUE_SCALE * D.MAX_TORQUE * (-u0 + u1 + u2 - u3))

    # --- semi-implicit Euler (Newton's integrate_rigid_body, centre of mass at the origin)
    v1 = v[k, b] + (force / D.MASS + wp.vec3(0.0, 0.0, D.GRAVITY)) * D.DT
    x1 = x[k, b] + v1 * D.DT
    inertia = wp.vec3(IXX, IYY, IZZ)
    w_b = wp.quat_rotate_inv(r, w[k, b])
    w_b = w_b + wp.cw_div(torque_b - wp.cross(w_b, wp.cw_mul(inertia, w_b)), inertia) * D.DT
    w1 = wp.quat_rotate(r, w_b)
    r1 = wp.normalize(r + wp.quat(w1, 0.0) * r * (0.5 * D.DT))
    w1 = w1 * (1.0 - D.ANGULAR_DAMPING * D.DT)
    x[k + 1, b] = x1
    q[k + 1, b] = r1
    v[k + 1, b] = v1
    w[k + 1, b] = w1

    # --- cost of this step (Newton's drone_cost + collision_cost)
    target = wp.vec3(TX, TY, TZ)
    altitude = wp.max(x1[2] - 0.75, 0.0) + wp.max(0.25 - x1[2], 0.0)
    upright = 1.0 - wp.quat_rotate(r1, wp.vec3(0.0, 0.0, 1.0))[2]
    c = (D.W_POS * wp.length_sq(x1 - target) + D.W_ALT * altitude
         + D.W_CONTROL * (u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3)
         + D.W_VEL * (wp.length_sq(v1) + wp.length_sq(w1)) + D.W_UP * upright)
    c = c * (D.COST_WEIGHT / D.W_TOTAL) * wp.pow(0.8, float(D.HORIZON - k - 1)) / float(D.HORIZON * D.HORIZON)
    local = x1 - wp.vec3(PX, PY, PZ)
    along = wp.clamp(local[2], -D.POLE_HALF_HEIGHT, D.POLE_HALF_HEIGHT)
    distance = wp.max(wp.length(local - wp.vec3(0.0, 0.0, along)) - D.POLE_RADIUS, 0.0)
    c = c + D.COLLISION_WEIGHT * wp.max(D.COLLISION_MARGIN - distance, 0.0)
    wp.atomic_add(cost, b, c)


class WarpRollout:
    """Allocates the per-step state slots once; ``value_and_grad`` replays the captured forward + backward graph."""

    def __init__(self, batch: int, device: str = "cuda:0"):
        H, self.device = D.HORIZON, device
        shape = (H + 1, batch)
        self.points = wp.zeros((batch, D.NUM_POINTS, 4), dtype=float, requires_grad=True, device=device)
        self.x = wp.zeros(shape, dtype=wp.vec3, requires_grad=True, device=device)
        self.q = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (H + 1, batch, 1)), dtype=wp.quat, requires_grad=True,
                          device=device)
        self.v = wp.zeros(shape, dtype=wp.vec3, requires_grad=True, device=device)
        self.w = wp.zeros(shape, dtype=wp.vec3, requires_grad=True, device=device)
        self.cost = wp.zeros(batch, dtype=float, requires_grad=True, device=device)
        self.arrays = (self.points, self.x, self.q, self.v, self.w, self.cost)
        self.forward_backward()                         # compile the kernels before capturing the graph
        with wp.ScopedCapture(device=device) as capture:
            self.forward_backward()
        self.graph = capture.graph

    def forward_backward(self):
        for a in self.arrays:                           # clear gradients and the cost (the start state is kept)
            a.grad.zero_()
        self.cost.zero_()
        tape = wp.Tape()
        with tape:
            for k in range(D.HORIZON):
                wp.launch(drone_step, dim=self.cost.shape[0], device=self.device,
                          inputs=(self.points, k, self.x, self.q, self.v, self.w), outputs=(self.cost,))
        self.cost.grad.fill_(1.0)                       # d(sum of costs) / d(cost_b) = 1
        tape.backward()

    def run(self):
        """One optimisation step's worth of work: replay the captured forward + backward graph."""
        wp.capture_launch(self.graph)

    def value_and_grad(self, points: np.ndarray):
        self.points.assign(points.astype(np.float32))
        self.run()
        wp.synchronize_device(self.device)
        return float(self.cost.numpy().sum()), self.points.grad.numpy()
