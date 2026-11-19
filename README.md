# Accelerating Higher-Order Automatic Differentiation for Sequential Scientific Computing

**Website:** https://thribhuvanrapolu.github.io/accelerating-higher-order-autodiff/

Many scientific models are long loops of small steps, such as a Kalman filter over a sensor log or a simulator rolled
out for model-predictive control. In natural JAX, each step becomes many tiny GPU kernels, and the runtime is set by
kernel-launch overhead rather than arithmetic. This repository runs the **whole loop in one Pallas kernel**, gets the
backward pass from a second kernel with **`jax.vjp` inside the kernel**, and connects the two with `jax.custom_vjp`.
Derivatives of derivatives (a Jacobian inside the loss) come out automatically.

Two experiments, each on an official example of another library:

| experiment | baseline | result (RTX 6000 Ada) |
|---|---|---|
| 1. Drone model-predictive control (Newton's `example_diffsim_drone.py`) | NVIDIA Warp / Newton | first-order gradient 0.056 ms vs. 0.162 ms (Warp) and 0.917 ms (Newton), for 16 rollouts. A second-order objective, which released Warp cannot differentiate automatically, takes 0.096 ms |
| 2. Learning a pendulum through an extended Kalman filter (Dynamax's EKF notebook) | Dynamax | 0.231 ms vs. 59.93 ms per gradient (259.9×), 3 kernels instead of 32,427 |

The full write-up (method, math, results, limitations) is on the website.

## Install

```bash
git clone https://github.com/thribhuvanrapolu/accelerating-higher-order-autodiff.git
cd accelerating-higher-order-autodiff
pip install -r requirements.txt     # JAX 0.8.2 (CUDA 12), Dynamax 1.0.3, Warp 1.19, Newton 1.8
```

Kernel counts also need NVIDIA Nsight Systems (`nsys`), which comes with the CUDA toolkit. Put it on your `PATH`.

## How to run

Run everything from the repository root, on an idle GPU:

```bash
export CUDA_VISIBLE_DEVICES=0            # choose an idle GPU
```

**Experiment 1: drone, vs. NVIDIA Warp** ([`example1_drone_vs_warp/`](example1_drone_vs_warp/README.md))

```bash
python -m example1_drone_vs_warp.check_correctness                  # first-order gradient vs. float64 (incl. Warp)
python -m example1_drone_vs_warp.uncertainty                        # second-order gradient vs. float64
python -m example1_drone_vs_warp.benchmark                          # timings, first order (incl. Newton's example)
python -m example1_drone_vs_warp.benchmark --uncertainty            # timings, second order
python -m example1_drone_vs_warp.benchmark --kernels                # nsys kernel counts, first order
python -m example1_drone_vs_warp.benchmark --uncertainty --kernels  # nsys kernel counts, second order
python -m example1_drone_vs_warp.make_figures
```

**Experiment 2: pendulum EKF, vs. Dynamax** ([`example2_pendulum_vs_dynamax/`](example2_pendulum_vs_dynamax/README.md))

```bash
python -m example2_pendulum_vs_dynamax.check_correctness            # all implementations vs. float64
python -m example2_pendulum_vs_dynamax.benchmark                    # timings
python -m example2_pendulum_vs_dynamax.benchmark --kernels --batch 1                  # nsys counts at B = 1
mv example2_pendulum_vs_dynamax/results/kernels.json example2_pendulum_vs_dynamax/results/kernels_b1.json
python -m example2_pendulum_vs_dynamax.benchmark --kernels                            # nsys counts at B = 64
python -m example2_pendulum_vs_dynamax.fit_gravity                  # learn g and r with Adam
python -m example2_pendulum_vs_dynamax.make_figures
```

Results are written to `results/*.json` and figures to `figures/*.png` inside each experiment folder.

## Repository layout

| folder / file | contents |
|---|---|
| [`common/`](common/) | `row_kernel.py` (generic fused kernel with in-kernel `jax.vjp`), `timing.py`, `nsys_count.py`, `plot_style.py` |
| [`example1_drone_vs_warp/`](example1_drone_vs_warp/README.md) | drone physics, JAX / Pallas / Warp implementations, second-order objective, benchmarks |
| [`example2_pendulum_vs_dynamax/`](example2_pendulum_vs_dynamax/README.md) | pendulum EKF: Dynamax / JAX / Pallas implementations, benchmarks, parameter learning |
| [`website.md`](website.md) | the website text (single source) |
| [`site/`](site/) | `build.py` and `template.html`, which turn `website.md` into `index.html` |

## Measurement setup

- NVIDIA RTX 6000 Ada (48 GB), driver 595.99.02, and an AMD Ryzen Threadripper PRO 9955WX CPU. Python 3.12, benchmarks
  run one after another on an otherwise idle GPU.
- All runs in FP32 with `jax_default_matmul_precision = "highest"` (no TF32). float64 is used only for the CPU
  reference in the correctness checks.
- Timings are median synchronized wall times after warm-up (20 GPU and 10 CPU evaluations for the pendulum, 50 GPU
  evaluations for the drone). First-call times are reported separately and include compilation.
- Kernel counts come from Nsight Systems 2026.1.3, in a separate capture of three warmed calls.

## Rebuilding the website

```bash
pip install markdown==3.7 pymdown-extensions==10.12
python site/build.py                 # website.md -> index.html
```

GitHub Pages serves `index.html` from the root of the `main` branch.

## License

Apache License 2.0, see [LICENSE](LICENSE). The drone physics and cost are ported from Newton's
`example_diffsim_drone.py` (Copyright (c) 2025 The Newton Developers, Apache License 2.0).

## Citation

```bibtex
@misc{rapolu2026higherorderad,
  title        = {Accelerating Higher-Order Automatic Differentiation for Sequential Scientific Computing},
  author       = {Rapolu, Thribhuvan},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://thribhuvanrapolu.github.io/accelerating-higher-order-autodiff/}},
  note         = {Code: \url{https://github.com/thribhuvanrapolu/accelerating-higher-order-autodiff}}
}
```
