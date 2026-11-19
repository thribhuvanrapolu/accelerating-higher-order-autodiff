# Example 1 — Drone model-predictive control through a differentiable simulator (vs. NVIDIA Warp)

We take the drone of NVIDIA's official differentiable-simulation example,
[`newton/examples/diffsim/example_diffsim_drone.py`](https://github.com/newton-physics/newton/tree/main/newton/examples/diffsim)
(Newton physics engine, built on Warp). A quadrotor must fly to a target next to a pole; its propeller commands are
improved by **gradient descent through the simulation** (model-predictive control, MPC).

* **Part A** is the example as NVIDIA wrote it: the cost contains no derivatives ($d = 0$), training needs first-order
  gradients. Warp is built for exactly this.
* **Part B** adds one realistic ingredient: an uncertainty term that is propagated with the simulator's Jacobian
  ($d = 1$). Training now needs a derivative of a derivative, which Warp's released autodiff does not provide.

## 1. The drone

One rigid body (mass $m = 0.56$ kg, inertia $I = \mathrm{diag}(0.00376, 0.00376, 0.00749)$ kg m²) with four
propellers at $p_i \in \{(0, \pm 0.2, 0), (\pm 0.2, 0, 0)\}$ m. The state is position $x$, orientation quaternion $q$
(rotation matrix $R(q)$), velocity $v$ and angular velocity $\omega$: 13 numbers.

**Controls.** Each propeller has 4 control points $c_{j,i}$ over the 30-step horizon (one every 10 steps). The
command at step $k$ is a linear interpolation, written with "hat" weights, times a gain of $0.8$:

$$u_i(k) = 0.8 \sum_{j=0}^{3} \max\!\big(0,\ 1 - |k/10 - j|\big)\, c_{j,i}.$$

**Forces.** Propeller $i$ pushes along the body $z$ axis with thrust $T u_i$ ($T = 4.18$ N) and twists with drag
torque $\pm M u_i$ ($M = 0.0556$ N m); Newton scales the torques by $0.8$:

$$F = R\, e_z\, T \sum_i u_i, \qquad \tau_b = 0.8 \Big(\sum_i p_i \times e_z\, T u_i \;+\; e_z\, M \sum_i s_i u_i\Big),$$

with spin directions $s = (-1, 1, 1, -1)$.

**One step** of Newton's `SolverSemiImplicit` ($\Delta t = 1/60$ s, gravity $g = -9.81\,e_z$, angular damping
$0.05$):

$$v_{k+1} = v_k + \Big(\frac{F}{m} + g\Big)\Delta t, \qquad x_{k+1} = x_k + v_{k+1}\,\Delta t,$$

$$\omega_b = R^\top\omega_k, \qquad \omega_b \leftarrow \omega_b + I^{-1}\big(\tau_b - \omega_b \times I\omega_b\big)\Delta t, \qquad \omega_{k+1} = (1 - 0.05\,\Delta t)\, R\,\omega_b,$$

$$q_{k+1} = \mathrm{normalise}\Big(q_k + \tfrac12\,\Delta t\, (\omega, 0) \otimes q_k\Big).$$

Code: [`drone.py`](drone.py) (`simulate_step`), a line-by-line port of Newton's `integrate_rigid_body` and
`compute_prop_wrenches`.

## 2. The cost (Part A)

After each step $k = 0 \dots 29$, Newton's `drone_cost` and `collision_cost`:

$$c_k = \frac{10^3}{\textstyle\sum w}\, \gamma_k \Big( 1000\,\lVert x - x^\star\rVert^2 + 100\, a(x_z) + 0.05\,\lVert u\rVert^2 + 0.1\,(\lVert v\rVert^2 + \lVert\omega\rVert^2) + 10\,(1 - e_z^\top R e_z) \Big) + 10^4 \max(0,\ 0.2 - d_{\text{pole}}),$$

with target $x^\star = (1, 0, 0.5)$, altitude penalty $a(z) = \max(z - 0.75, 0) + \max(0.25 - z, 0)$, discount
$\gamma_k = 0.8^{29 - k}/30^2$, and $d_{\text{pole}}$ the distance from the drone to a vertical capsule (radius 0.15 m).
MPC needs

$$\nabla_c L, \qquad L = \sum_{b=1}^{B} \sum_{k=0}^{29} c_k^{(b)},$$

for $B$ rollouts (the example uses $B = 16$), every optimisation step.

## 3. Part B: uncertainty-aware cost (a Jacobian inside the cost)

The drone does not know its starting tilt exactly. A small tilt error $\delta_0$ (0.05 rad about $x$, $y$ or $z$)
travels along the rollout through the **linearised** dynamics:

$$\delta_{k+1} = F_k\,\delta_k, \qquad F_k = \frac{\partial\, \mathrm{step}}{\partial\, \mathrm{state}}\Big|_{(\mathrm{state}_k,\ u_k)}.$$

This is the covariance propagation of an extended Kalman filter, $P_{k+1} = F_k P_k F_k^\top$ with
$P_0 = \sum_j \delta_{0j}\delta_{0j}^\top$. We penalise the spread of the position:

$$c_k \mathrel{+}= 100\, \operatorname{tr}\big(P_k[x, x]\big) = 100 \sum_{j=1}^{3} \lVert \delta_{kj}[x] \rVert^2 .$$

Now $\nabla_c L$ needs $\partial F_k / \partial c$: **the derivative of a Jacobian**, second-order AD through 30 steps.

* JAX: `jax.jacfwd` (natural code) or `jax.jvp` (per-row code); `jax.grad` differentiates it again automatically.
* Pallas: one row per (rollout $b$, error direction $j$). Each row carries the state and **one** tangent
  $\delta_{kj}$ and advances both with a single `jax.jvp` inside the forward kernel; the backward kernel runs
  `jax.vjp` of it, so a second-order derivative is generated inside the kernel. Code: [`uncertainty.py`](uncertainty.py).

  *Lesson learned:* our first version carried all three tangents in one row. Its backward kernel was about three
  times bigger, and Triton had not finished compiling it after 15 minutes (Triton's compile time grows roughly
  quadratically with kernel size). One tangent per row: 15.7 s for the first call, compile included.
* Warp (released 1.19): reverse mode only; `wp.Tape` cannot differentiate a gradient or Jacobian again. You must
  hand-write $F_k\,\delta$ for this step (quaternion product, normalisation, gyroscopic term, …) and let Warp
  differentiate that. Warp's development branch has *experimental* forward-mode "jets"
  (`wp.JetSpace`) that may make this possible in a future release.

## 4. The implementations

| file | implementation | how the loop runs on the GPU |
|---|---|---|
| [`rollout_jax.py`](rollout_jax.py) `loss_natural` | natural JAX: small vectors, `vmap`, `lax.scan` | XLA: hundreds of small kernels per step |
| [`rollout_jax.py`](rollout_jax.py) `loss_lanes` | per-row JAX (each scalar is an array over rollouts) | XLA, fewer kernels |
| [`rollout_pallas.py`](rollout_pallas.py) | **one Pallas kernel for the whole rollout**, one for its backward pass (`jax.vjp` inside) | 2 kernels |
| [`rollout_warp.py`](rollout_warp.py) | Warp the Warp way: one kernel per step, `wp.Tape`, CUDA graph | 30 forward + 30 adjoint kernels |
| Newton's own example | the official code, timed through its captured CUDA graph | ~390 kernels |

All are checked against a float64 reference in [`check_correctness.py`](check_correctness.py) (cost and gradient
agree to ~1e-6 relative).

```bash
cd jax_pallas
python -m example1_drone_vs_warp.check_correctness       # Part A, incl. Warp
python -m example1_drone_vs_warp.uncertainty             # Part B correctness
python -m example1_drone_vs_warp.benchmark               # Part A timings (incl. Newton's example)
python -m example1_drone_vs_warp.benchmark --uncertainty # Part B timings
python -m example1_drone_vs_warp.benchmark --kernels     # nsys kernel counts
python -m example1_drone_vs_warp.make_figures
```
