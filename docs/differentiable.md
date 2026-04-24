# Differentiable Workflows

`neurosim` is built so the deterministic simulation path can be used in
gradient-based workflows. The important distinction is between:

- deterministic solvers that are differentiable through JAX
- stochastic sampling routines that are useful for Monte Carlo studies
  but are not generally the right object for gradient descent

## What Is Differentiable

These parts of the library are intended for automatic differentiation:

- `neurosim.classical` systems built from smooth equations of motion
- `neurosim.optimize.optimize`
- `neurosim.optimize.sensitivity`
- `neurosim.optimize.parameter_sweep`
- `neurosim.optimize.refine_parameter_sweep`
- `neurosim.quantum.solve_schrodinger`
- `neurosim.quantum.solve_eigenvalue_problem`
- `neurosim.optics` routines that map parameters to smooth field values

## Typical Pattern

1. Write a simulation function that maps inputs to a scalar objective.
2. Use `neurosim.parameter_sweep` to map the search space when a coarse
   global scan is useful.
3. Pass the best basin to `jax.grad` or `neurosim.optimize.optimize`.
4. Keep the objective scalar and use JAX arrays throughout.

Example:

```python
import jax
import jax.numpy as jnp
import neurosim as ns

def miss_distance(v0):
    return (ns.projectile(v0=v0, angle=35.0, g=1.62).range - 1200.0) ** 2

grad = jax.grad(miss_distance)
print(float(grad(200.0)))
```

For grid search before gradient descent, build a named Cartesian grid and
rank it with a scalar objective:

```python
import jax.numpy as jnp
import neurosim as ns

grid = ns.make_parameter_grid(
    {
        "v0": jnp.linspace(20.0, 60.0, 21),
        "angle": jnp.linspace(25.0, 65.0, 17),
    }
)

target = 120.0
result = ns.parameter_sweep(
    lambda params: ns.projectile(v0=params[0], angle=params[1]).range,
    grid.values,
    objective=lambda range_m: (range_m - target) ** 2,
    batch_size=64,
)

print(grid.as_dict(result.best_index))
print(result.summary())
```

`parameter_sweep` uses `jax.vmap` by default, supports chunked evaluation with
`batch_size`, and can maximize scores by passing `minimize=False`.

When the grid identifies a promising basin but not a precise optimum, refine
the best sweep seeds with local gradient descent:

```python
import jax.numpy as jnp
import neurosim as ns

grid = ns.make_parameter_grid(
    {
        "v0": jnp.linspace(20.0, 60.0, 21),
        "angle": jnp.linspace(25.0, 65.0, 17),
    }
)

target = 120.0

def objective(params):
    return (ns.projectile(v0=params[0], angle=params[1]).range - target) ** 2

sweep = ns.parameter_sweep(objective, grid.values, batch_size=64)
refined = ns.refine_parameter_sweep(
    objective,
    sweep,
    top_k=3,
    learning_rate=0.01,
    max_iterations=500,
)

print(refined.best_parameters)
print(refined.summary())
```

## Caveats

- Randomized Monte Carlo updates, such as Ising sampling, are meant for
  statistics and visualization, not gradient descent.
- `refine_parameter_sweep` requires a minimization sweep, so negate the reward
  objective before sweeping and refining a maximization search.
- Long-running simulations may still need parameter tuning for stable
  gradients, especially when the objective is poorly conditioned.
- `jax_enable_x64` is enabled at import time to improve numerical
  stability for physics workloads.
