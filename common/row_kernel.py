"""Turn a small per-row JAX function into ONE fused, differentiable GPU kernel (Pallas, Triton backend).

The idea in three lines
-----------------------
1. Write the math of one step for ONE problem, but with every scalar replaced by a vector over many problems
   ("rows"). A 2-number state becomes a tuple of 2 arrays of shape (rows,). We call each array a "lane".
2. The forward kernel runs that function on a block of rows. All the little operations (sin, *, +, ...) happen in
   registers inside one kernel, instead of one tiny kernel each.
3. The backward kernel runs ``jax.vjp`` of the SAME function inside the kernel. JAX writes the derivative code for
   us; ``jax.custom_vjp`` tells ``jax.grad`` to use this kernel. No hand-written derivatives anywhere.

Data layout
-----------
Inputs and outputs are 2-D arrays of shape (lanes, rows): row i of the array is lane i, column j is problem j.
A kernel program handles ``block`` problems (columns) at a time.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl


def make_row_op(step_fn, n_in: int, n_out: int, block: int = 32):
    """Return ``op(x) -> y`` with x of shape (n_in, N) and y of shape (n_out, N), computed by one Pallas kernel.

    ``step_fn(lanes) -> lanes`` takes a tuple of ``n_in`` arrays of shape (block,) and returns a tuple of ``n_out``
    arrays of the same shape. It may use any jax.numpy math, and even jax.jvp / jax.grad inside.
    N must be a multiple of ``block``. ``op`` can be used inside jax.grad / jax.value_and_grad.
    """

    def columns(height):
        # program i sees columns [i * block, (i + 1) * block) of every lane
        return pl.BlockSpec((height, block), lambda i: (0, i))

    # ----------------------------------------------------------------------------------- the two kernels
    def forward_kernel(x_ref, y_ref):
        x = tuple(x_ref[k, :] for k in range(n_in))          # load each lane (Triton cannot slice registers)
        y = step_fn(x)
        for k in range(n_out):
            y_ref[k, :] = y[k]

    def backward_kernel(x_ref, y_bar_ref, x_bar_ref):
        x = tuple(x_ref[k, :] for k in range(n_in))
        _, pullback = jax.vjp(step_fn, x)                     # autodiff INSIDE the kernel
        (x_bar,) = pullback(tuple(y_bar_ref[k, :] for k in range(n_out)))
        for k in range(n_in):
            x_bar_ref[k, :] = x_bar[k]

    # ----------------------------------------------------------------------------------- launching them
    def run_forward(x):
        return pl.pallas_call(
            forward_kernel,
            grid=(x.shape[1] // block,),
            in_specs=[columns(n_in)],
            out_specs=columns(n_out),
            out_shape=jax.ShapeDtypeStruct((n_out, x.shape[1]), x.dtype),
        )(x)

    def run_backward(x, y_bar):
        return pl.pallas_call(
            backward_kernel,
            grid=(x.shape[1] // block,),
            in_specs=[columns(n_in), columns(n_out)],
            out_specs=columns(n_in),
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y_bar)

    # ----------------------------------------------------------------------------------- tell JAX how to differentiate
    @jax.custom_vjp
    def op(x):
        return run_forward(x)

    def op_fwd(x):
        return run_forward(x), x                              # remember only the input ...

    def op_bwd(x, y_bar):
        return (run_backward(x, y_bar),)                      # ... and recompute the step inside the backward kernel

    op.defvjp(op_fwd, op_bwd)
    return op
