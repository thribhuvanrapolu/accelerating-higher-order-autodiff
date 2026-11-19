"""Small helpers shared by both examples: precision, devices, timing and the nsys profiler range."""

from __future__ import annotations

import ctypes
import glob
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # several processes may share a GPU

import jax
import numpy as np

# True FP32 everywhere: on an A100, JAX's default matrix products use TF32 (about 3 significant digits).
jax.config.update("jax_default_matmul_precision", "highest")


def time_call(fn, *args, repeats: int = 20) -> tuple[float, float]:
    """(median seconds per call, seconds for the first call = compile + run). Waits for the GPU every call."""
    tic = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - tic
    times = []
    for _ in range(repeats):
        tic = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - tic)
    return float(np.median(times)), first


class profiler_range:
    """``with profiler_range(): ...`` marks the region that nsys records (run nsys with
    ``--capture-range=cudaProfilerApi``). Outside nsys it does nothing harmful."""

    def __enter__(self):
        paths = glob.glob(os.path.join(__import__("nvidia").__path__[0], "cuda_runtime/lib/libcudart.so*"))
        self.cudart = ctypes.CDLL(paths[0])
        self.cudart.cudaProfilerStart()

    def __exit__(self, *exc):
        self.cudart.cudaProfilerStop()
