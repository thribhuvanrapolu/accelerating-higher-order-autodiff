"""Count the GPU kernels a piece of code launches, with NVIDIA Nsight Systems (nsys).

The profiled script must wrap the calls it wants counted in ``common.timing.profiler_range()`` and repeat them
``calls`` times. We run it under nsys and read the kernel summary:

    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -t cuda  python -m ...
    nsys stats --report cuda_gpu_kern_sum --format csv  report.nsys-rep

``--cuda-graph-trace=node`` matters: XLA records some kernels into CUDA graphs, and without it nsys shows a graph as
one item instead of the kernels inside it.
"""

from __future__ import annotations

import csv
import io
import os
import pathlib
import subprocess
import tempfile


def count_kernels(module_args: list[str], calls: int, cwd: str, env: dict | None = None) -> dict:
    """Run ``python -m <module_args>`` under nsys; return kernels and GPU-busy time per call."""
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "report"
        subprocess.run(
            ["nsys", "profile", "-o", str(report), "--force-overwrite", "true", "--capture-range=cudaProfilerApi",
             "--capture-range-end=stop",            # let the script finish normally after the recorded region
             "--cuda-graph-trace=node", "-t", "cuda", "python", "-m", *module_args],
            cwd=cwd, env={**os.environ, **(env or {})}, check=True, capture_output=True)
        text = subprocess.run(
            ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", f"{report}.nsys-rep"],
            check=True, capture_output=True, text=True).stdout
    rows = [r for r in csv.reader(io.StringIO(text)) if len(r) > 4]
    header, body = rows[0], rows[1:]
    instances, total_ns = header.index("Instances"), header.index("Total Time (ns)")
    kernels = sum(int(r[instances]) for r in body)
    busy = sum(float(r[total_ns]) for r in body)
    return {"kernels_per_call": kernels / calls, "gpu_busy_ms_per_call": busy / calls / 1e6,
            "mean_kernel_us": busy / max(kernels, 1) / 1e3}
