# GPU Acceleration

`neurosim` uses JAX as its execution backend, so the same code can run
on CPU or GPU depending on which `jaxlib` build is installed in the
environment.

## How It Works

- Array-heavy kernels are written in JAX-friendly form.
- Iterative solvers use `jax.lax.scan` or vectorized array updates where
  possible.
- The package enables 64-bit mode on import to keep physical units and
  long-time integration stable.

## What To Expect On GPU

When JAX is configured for CUDA or ROCm, the following classes of
workload benefit the most:

- N-body gravity
- FDTD field updates
- Quantum time evolution
- Batched temperature sweeps and Monte Carlo statistics

The speedup depends on problem size. Small examples may run faster on
CPU because the JIT compile overhead dominates.

## Setup

Install a GPU-enabled JAX build that matches your platform. Then run
the examples or tests normally; `neurosim` does not need a separate GPU
flag.

## Practical Advice

- Use larger lattices or longer trajectories if you want to measure GPU
  throughput meaningfully.
- Keep the first call separate from timed runs because JAX compiles on
  first execution.
- Prefer JAX arrays throughout your own code to avoid host-device
  transfers.
