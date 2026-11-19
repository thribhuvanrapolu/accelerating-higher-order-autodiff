"""Time one training step (loss + gradient) of every implementation, and count its GPU kernels.

    cd jax_pallas
    python -m example2_pendulum_vs_dynamax.benchmark                 # timings  -> results/timings.json
    python -m example2_pendulum_vs_dynamax.benchmark --kernels       # nsys     -> results/kernels.json

One "training step" = jax.jit(jax.value_and_grad(loss)) on B measurement sequences of T = 400 steps.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from common.timing import profiler_range, time_call  # (importing common.timing also sets FP32 precision)
import jax
import jax.numpy as jnp

from .check_correctness import IMPLEMENTATIONS
from .pendulum import simulate

HERE = pathlib.Path(__file__).parent
BATCHES = (1, 16, 256, 4096)
CPU_RUNS = ("Dynamax", "JAX natural")            # also timed on the CPU, for comparison


def training_step(name):
    return jax.jit(jax.value_and_grad(IMPLEMENTATIONS[name]))


def time_one(name, batch, device):
    with jax.default_device(jax.devices(device)[0]):
        ys = simulate(jax.random.PRNGKey(0), batch)
        theta = (jnp.float32(8.0), jnp.float32(0.4))
        seconds, first = time_call(training_step(name), theta, ys, repeats=10 if device == "cpu" else 20)
    return {"implementation": name, "device": device, "batch": batch, "ms": seconds * 1e3, "first_call_s": first}


def profile_one(name, batch, calls=3):
    """Body of the nsys run: warm up (compile), then repeat the call ``calls`` times inside the profiler range."""
    ys = simulate(jax.random.PRNGKey(0), batch)
    theta = (jnp.float32(8.0), jnp.float32(0.4))
    step = training_step(name)
    jax.block_until_ready(step(theta, ys))
    with profiler_range():
        for _ in range(calls):
            jax.block_until_ready(step(theta, ys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", action="store_true", help="count kernels with nsys (needs nsys on PATH)")
    parser.add_argument("--profile", help=argparse.SUPPRESS)              # internal: the run nsys records
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)

    if args.profile:
        profile_one(args.profile, args.batch)
        return

    if args.kernels:
        from common.nsys_count import count_kernels
        results = []
        for name in IMPLEMENTATIONS:
            counts = count_kernels([f"{__package__}.benchmark", "--profile", name, "--batch", str(args.batch)],
                                   calls=3, cwd=str(HERE.parent))
            results.append({"implementation": name, "batch": args.batch, **counts})
            print(f"{name:26s} {counts['kernels_per_call']:9.0f} kernels/step  "
                  f"({counts['kernels_per_call'] / 400:6.1f} per EKF step)  mean {counts['mean_kernel_us']:.2f} us",
                  flush=True)
        (out / "kernels.json").write_text(json.dumps(results, indent=2))
        return

    results = []
    for batch in BATCHES:
        for name in IMPLEMENTATIONS:
            for device in ("gpu", "cpu") if name in CPU_RUNS else ("gpu",):
                if device == "cpu" and batch > 256:
                    continue                                     # minutes per step on the CPU: skipped
                r = time_one(name, batch, device)
                results.append(r)
                print(f"B={batch:5d} {device} {name:26s} {r['ms']:9.2f} ms   (first call {r['first_call_s']:.1f} s)",
                      flush=True)
    (out / "timings.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
