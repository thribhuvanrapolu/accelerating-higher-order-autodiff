"""Time one MPC optimisation step (cost + gradient w.r.t. the control points) of every implementation.

    cd jax_pallas
    python -m example1_drone_vs_warp.benchmark              # timings  -> results/timings.json
    python -m example1_drone_vs_warp.benchmark --kernels    # nsys kernel counts (B = 16) -> results/kernels.json
    python -m example1_drone_vs_warp.benchmark --uncertainty [--kernels]   # Part B (cost with a Jacobian inside)

Implementations: JAX natural, JAX lanes, Pallas whole rollout, our Warp version, and NVIDIA's official Newton
example itself (its captured forward + backward graph, same drone, same horizon, same number of rollouts).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from common.timing import profiler_range, time_call  # (also sets FP32 precision)
import jax
import numpy as np

from .check_correctness import JAX_IMPLEMENTATIONS, control_points

HERE = pathlib.Path(__file__).parent
BATCHES = (16, 256, 4096)
NAMES = (*JAX_IMPLEMENTATIONS, "Warp (ours)", "Newton example")


def make_runner(name, batch, part="a"):
    """A function that does one optimisation step's worth of work and waits for the GPU."""
    points = control_points(batch)
    if part == "b":                                     # Part B: the uncertainty-aware cost (JAX versions only)
        from .uncertainty import IMPLEMENTATIONS
        step = jax.jit(jax.value_and_grad(IMPLEMENTATIONS[name]))
        return lambda: jax.block_until_ready(step(points))
    if name in JAX_IMPLEMENTATIONS:
        step = jax.jit(jax.value_and_grad(JAX_IMPLEMENTATIONS[name]))
        return lambda: jax.block_until_ready(step(points))
    import warp as wp
    if name == "Warp (ours)":
        from .rollout_warp import WarpRollout
        rollout = WarpRollout(batch)
        rollout.points.assign(np.asarray(points))
        return lambda: (rollout.run(), wp.synchronize_device())
    if name == "Newton example":
        import newton.viewer
        from newton.examples.diffsim.example_diffsim_drone import Example
        args = Example.create_parser().parse_args(["--viewer", "null", "--num_rollouts", str(batch)])
        example = Example(newton.viewer.ViewerNull(), args)
        return lambda: (wp.capture_launch(example.graph), wp.synchronize_device())
    raise ValueError(name)


def time_runner(run, repeats=50):
    tic = time.perf_counter()
    run()
    first = time.perf_counter() - tic
    ms, _ = time_call(lambda: run(), repeats=repeats)
    return ms * 1e3, first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--uncertainty", action="store_true", help="Part B: cost with the propagated uncertainty")
    args = parser.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    part, suffix = ("b", "_uncertainty") if args.uncertainty else ("a", "")
    if args.uncertainty:
        from .uncertainty import IMPLEMENTATIONS
        names = tuple(IMPLEMENTATIONS)
    else:
        names = NAMES

    if args.profile:                                   # the run that nsys records
        run = make_runner(args.profile, args.batch, part)
        run()
        with profiler_range():
            for _ in range(3):
                run()
        return

    if args.kernels:
        from common.nsys_count import count_kernels
        results = []
        for name in names:
            extra = ["--uncertainty"] if args.uncertainty else []
            counts = count_kernels([f"{__package__}.benchmark", "--profile", name, "--batch", str(args.batch), *extra],
                                   calls=3, cwd=str(HERE.parent))
            results.append({"implementation": name, "batch": args.batch, **counts})
            print(f"{name:24s} {counts['kernels_per_call']:8.0f} kernels/step  mean {counts['mean_kernel_us']:.2f} us"
                  f"  GPU busy {counts['gpu_busy_ms_per_call']:.3f} ms", flush=True)
        (out / f"kernels{suffix}.json").write_text(json.dumps(results, indent=2))
        return

    results = []
    for batch in BATCHES:
        for name in names:
            ms, first = time_runner(make_runner(name, batch, part))
            results.append({"implementation": name, "batch": batch, "ms": ms, "first_call_s": first})
            print(f"B={batch:5d} {name:24s} {ms:8.3f} ms   (first call {first:.1f} s)", flush=True)
    (out / f"timings{suffix}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
