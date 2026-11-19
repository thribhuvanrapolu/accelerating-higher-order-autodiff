# Example 2 — Learning a pendulum through an extended Kalman filter (vs. Dynamax)

We take the pendulum from [Dynamax](https://github.com/probml/dynamax)'s own EKF notebook
(`docs/notebooks/nonlinear_gaussian_ssm/ekf_ukf_pendulum.ipynb`, Särkkä's *Bayesian Filtering and Smoothing*,
Example 3.7). We go one step past the notebook: instead of only filtering, we **learn** the gravity $g$ and the
sensor noise $r$ from noisy measurements by gradient descent on the EKF log-likelihood.

## 1. The model

The state is the angle and the angular velocity, $x = [\alpha,\ \dot\alpha]$. With time step $\Delta t = 0.0125$ s:

$$x_{k+1} = f(x_k) + q_k, \qquad f(x) = \begin{bmatrix} \alpha + \dot\alpha\,\Delta t \\ \dot\alpha - g \sin\alpha\,\Delta t \end{bmatrix}, \qquad q_k \sim \mathcal N(0, Q),$$

$$y_k = h(x_k) + r_k, \qquad h(x) = \sin\alpha, \qquad r_k \sim \mathcal N(0, r^2),$$

with the process noise of a continuous white-noise acceleration of intensity $q_c = 1$:

$$Q = q_c \begin{bmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{bmatrix}.$$

True values: $g = 9.8$, $r = 0.3$. Data: $T = 400$ steps (5 s) per sequence, start $x_0 = [\pi/2,\ 0]$.
Code: [`pendulum.py`](pendulum.py).

## 2. The extended Kalman filter, one step

The filter keeps a Gaussian belief $\mathcal N(m_k, P_k)$ about the state. At every step (Dynamax's order: measure,
update, predict), with the Jacobians $H = \partial h/\partial x$ and $F = \partial f/\partial x$:

| | formula | meaning |
|---|---|---|
| innovation variance | $S = H P H^\top + r^2$ | how uncertain the next measurement is |
| innovation | $e = y - h(m)$ | measurement minus prediction |
| log-likelihood | $\ell = -\tfrac12\left(\log 2\pi S + e^2 / S\right)$ | how well the model predicted $y$ |
| gain | $K = P H^\top / S$ | how much to trust the measurement |
| update | $m \leftarrow m + K e$, $\;P \leftarrow P - K S K^\top$ | belief after seeing $y$ |
| predict | $m \leftarrow f(m)$, $\;P \leftarrow F P F^\top + Q$ | belief for the next step |

For the pendulum, $H = [\cos\alpha,\ 0]$ and

$$F = \begin{bmatrix} 1 & \Delta t \\ -g\cos\alpha\,\Delta t & 1 \end{bmatrix}.$$

We never write these Jacobians by hand: the code gets them from `jax.jacfwd` (natural form) or `jax.jvp`
(per-row form), exactly as Dynamax does (`F, H = jacfwd(f), jacfwd(h)` in
`dynamax/nonlinear_gaussian_ssm/inference_ekf.py`).

## 3. The training loss, and why it needs second derivatives

The loss is the negative log-likelihood per measurement, averaged over $B$ sequences:

$$L(g, r) = -\frac{1}{B\,T} \sum_{b=1}^{B} \sum_{k=1}^{T} \ell_k^{(b)}(g, r).$$

$\ell_k$ depends on $S_k$, which depends on $P_k$, which depends on every earlier $F_j$. The gradient
$\partial L / \partial g$ therefore contains $\partial F_j / \partial g$: **the derivative of a Jacobian**. In
derivative-counting terms, one derivative is *inside* the forward computation ($d = 1$), so training needs
automatic differentiation of order $d + 1 = 2$, through a loop of $T = 400$ sequential steps.

## 4. Why it is slow on a GPU

Each EKF step is a handful of operations on 2-vectors and 2×2 matrices: a few hundred floating-point operations.
XLA compiles `lax.scan` into a GPU loop whose body is still many small kernels. Nsight Systems counts on RTX 6000 Ada for one training step
(B = 64, T = 400):

| implementation | kernels per training step | per EKF step |
|---|---|---|
| Dynamax | 41,635 | 104.09 |
| JAX natural | 18,422 | 46.05 |
| JAX lanes | 11,212 | 28.03 |
| Pallas: kernel per step | 7,612 | 19.03 |
| Pallas: whole loop | 9 | 0.02 |

Dynamax averaged 104.1 kernels per EKF step. Nsight recorded 1.00 µs per kernel on average and 41.70 ms of summed kernel duration per gradient. The unprofiled wall times below also include dispatch and synchronization.

Dynamax GPU time rose from 59.93 ms at B = 1 to 72.80 ms at B = 4,096. The CPU took 3.39 ms for one sequence. The whole-loop GPU kernel sped up the Dynamax GPU evaluation by 259.9× and 251.9× at those batch sizes.

Median wall time in milliseconds per loss and gradient.

| B | Dynamax GPU | Dynamax CPU | JAX natural GPU | JAX natural CPU | JAX lanes GPU | Pallas per step | **Pallas whole loop** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59.93 | 3.39 | 20.21 | 1.33 | 13.54 | 6.90 | 0.231 |
| 16 | 65.30 | 8.15 | 21.39 | 1.43 | 12.76 | 6.96 | 0.274 |
| 256 | 63.97 | 84.13 | 22.79 | 11.51 | 14.23 | 7.84 | 0.282 |
| 4096 | 72.80 | – | 27.15 | – | 15.05 | 8.57 | 0.289 |

| Implementation | Device | Wall time for 300 steps | Final $(g, r)$ |
|---|---|---:|---|
| Dynamax | GPU | 19.37 s | (9.5044, 0.2991) |
| JAX natural | CPU | 0.542 s | (9.5044, 0.2991) |
| Pallas: whole loop | GPU | 0.098 s | (9.5044, 0.2991) |

## 5. The fix: the whole filter in one kernel, with autodiff inside the kernel

1. **Per-row layout** ([`ekf_jax.py`](ekf_jax.py), `ekf_step_lanes`). Write one EKF step with every scalar
   replaced by an array over sequences. The 2×2 covariance becomes three arrays $(p_{00}, p_{01}, p_{11})$.
2. **Forward kernel** ([`ekf_pallas.py`](ekf_pallas.py), `make_filter_op`). A `fori_loop` over the 400 steps
   *inside* one Pallas kernel. Each step's starting state is written to memory for the backward pass.
3. **Backward kernel**. A second kernel walks the steps backwards. At step $k$ it runs `jax.vjp(ekf_step_lanes)`
   *inside the kernel*, which turns the cotangent of step $k+1$'s state into the cotangent of step $k$'s state, and
   adds up $\partial L / \partial g$ and $\partial L / \partial r$. Because `ekf_step_lanes` itself calls `jax.jvp` for
   $F$ and $H$, the backward kernel contains JAX-generated second-order code; nothing is written by hand.
4. **`jax.custom_vjp`** connects the two kernels, so `jax.grad(loss)` just works.

## 6. Files

| file | what it does |
|---|---|
| [`pendulum.py`](pendulum.py) | the model and the data simulator |
| [`ekf_dynamax.py`](ekf_dynamax.py) | the loss computed with Dynamax (baseline) |
| [`ekf_jax.py`](ekf_jax.py) | the same loss in plain JAX: natural form, and per-row form |
| [`ekf_pallas.py`](ekf_pallas.py) | Pallas: one kernel per step, and the whole filter in one kernel |
| [`check_correctness.py`](check_correctness.py) | all implementations vs. a float64 reference |
| [`benchmark.py`](benchmark.py) | time per training step vs. batch size; nsys kernel counts |
| [`fit_gravity.py`](fit_gravity.py) | learn $g$ and $r$ with Adam, with three implementations |
| [`make_figures.py`](make_figures.py) | the figures in `figures/` |

```bash
# Run from the repository root
python -m example2_pendulum_vs_dynamax.check_correctness
python -m example2_pendulum_vs_dynamax.benchmark
python -m example2_pendulum_vs_dynamax.benchmark --kernels      # needs nsys on PATH
python -m example2_pendulum_vs_dynamax.fit_gravity
python -m example2_pendulum_vs_dynamax.make_figures
```
