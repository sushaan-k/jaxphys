"""3D electromagnetic dipole radiation using the full FDTD solver.

Demonstrates the 3D FDTD solver by simulating radiation from a
point dipole source. The fields propagate spherically outward,
showing the characteristic dipole radiation pattern.

The far-field radiation pattern of a z-oriented dipole has:
    E_theta ~ sin(theta)  (donut pattern)
"""

import jax.numpy as jnp

import neurosim as ns

# Grid parameters
n = 40  # grid cells per dimension
resolution = 0.01  # 1 cm cells

# Create 3D grid
grid = ns.EMGrid3D(
    size=(n, n, n),
    resolution=resolution,
    boundary="absorbing",
    pml_layers=6,
)

# Add a z-polarized point source at the center
center = n // 2
grid.add_source(
    ns.PointSource3D(
        frequency=3e9,  # 3 GHz (lambda ~ 10 cm)
        position=(center, center, center),
        amplitude=1.0,
        polarization="z",
    )
)

print(f"3D FDTD simulation: {n}x{n}x{n} grid")
print(f"Resolution: {resolution*100:.1f} cm")
print(f"Source: 3 GHz z-dipole at center")
print("Running simulation...")

# Run simulation
fields = grid.simulate(t_span=(0, 3e-9), save_every=10)

print(f"Completed: {fields.t.shape[0]} snapshots")

# Analyze the radiation pattern in the xy-plane (z = center)
ez_mid = fields.ez[-1, :, :, center]
max_field = float(jnp.max(jnp.abs(ez_mid)))
print(f"Max |Ez| in midplane: {max_field:.4e} V/m")

# Check spherical spreading: field should be present in all directions
# from the source (not just along one axis)
quadrants = [
    float(jnp.max(jnp.abs(ez_mid[:center, :center]))),
    float(jnp.max(jnp.abs(ez_mid[center:, :center]))),
    float(jnp.max(jnp.abs(ez_mid[:center, center:]))),
    float(jnp.max(jnp.abs(ez_mid[center:, center:]))),
]
print(f"Field in quadrants: {[f'{q:.4e}' for q in quadrants]}")

# Total energy in the grid (sum of E^2)
total_E2 = float(
    jnp.sum(fields.ex[-1] ** 2 + fields.ey[-1] ** 2 + fields.ez[-1] ** 2)
)
total_H2 = float(
    jnp.sum(fields.hx[-1] ** 2 + fields.hy[-1] ** 2 + fields.hz[-1] ** 2)
)
print(f"Total |E|^2: {total_E2:.4e}")
print(f"Total |H|^2: {total_H2:.4e}")

# Also demonstrate a dielectric slab
print("\n--- With dielectric slab (eps_r=4) ---")
grid2 = ns.EMGrid3D(
    size=(n, n, n), resolution=resolution, boundary="absorbing", pml_layers=6
)
grid2.add_source(
    ns.PointSource3D(frequency=3e9, position=(center, center, center))
)

# Add dielectric slab on one side
mask = jnp.zeros((n, n, n), dtype=bool)
mask = mask.at[center + 5 :, :, :].set(True)
grid2.add_material(ns.DielectricRegion(mask=mask, epsilon_r=4.0))

fields2 = grid2.simulate(t_span=(0, 3e-9), save_every=10)
ez_mid2 = fields2.ez[-1, :, :, center]

# Compare: the dielectric should slow propagation on one side
free_side = float(jnp.max(jnp.abs(ez_mid2[:center, :])))
diel_side = float(jnp.max(jnp.abs(ez_mid2[center:, :])))
print(f"Max field (free space side): {free_side:.4e}")
print(f"Max field (dielectric side): {diel_side:.4e}")
