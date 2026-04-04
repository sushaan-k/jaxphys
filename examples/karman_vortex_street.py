"""Karman vortex street behind a cylinder using Lattice Boltzmann.

Demonstrates the LBM solver by simulating flow past a cylindrical
obstacle, producing the characteristic alternating vortex pattern
(von Karman vortex street) at moderate Reynolds numbers.

The Reynolds number is:
    Re = U * D / nu

where U is the inlet velocity, D is the cylinder diameter, and
nu is the kinematic viscosity.
"""

import jax.numpy as jnp

import neurosim as ns

# Grid and flow parameters
nx, ny = 300, 100
viscosity = 0.02
u_inlet = 0.06
cylinder_x, cylinder_y = 60, 50
cylinder_r = 12

# Reynolds number: Re = U * D / nu
Re = u_inlet * (2 * cylinder_r) / viscosity
print(f"Reynolds number: Re = {Re:.1f}")

# Create the LBM grid
grid = ns.LBMGrid(size=(nx, ny), viscosity=viscosity)

# Create cylindrical obstacle
x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
cylinder_mask = (x - cylinder_x) ** 2 + (y - cylinder_y) ** 2 < cylinder_r**2
grid.add_obstacle(ns.Obstacle(mask=cylinder_mask))

print(f"Grid: {nx} x {ny}, viscosity: {viscosity}")
print(f"Inlet velocity: {u_inlet}, tau: {grid.tau:.3f}")
print("Running LBM simulation...")

# Run simulation
result = grid.simulate(n_steps=10000, u_inlet=u_inlet, save_every=200)

print(f"Completed: {result.n_snapshots} snapshots saved")
print(f"Final max speed: {float(jnp.max(result.speed[-1])):.4f}")
print(f"Final max vorticity: {float(jnp.max(jnp.abs(result.vorticity[-1]))):.6f}")

# Check for vortex shedding: vorticity should be nonzero behind cylinder
wake_vorticity = result.vorticity[-1, cylinder_x + 20 :, :]
max_wake_vort = float(jnp.max(jnp.abs(wake_vorticity)))
print(f"Wake vorticity: {max_wake_vort:.6f}")
if max_wake_vort > 1e-6:
    print("Vortex shedding detected!")
