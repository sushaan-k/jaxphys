"""Plain NumPy reference implementations of the jaxphys solvers.

Each function implements the same numerical scheme as the corresponding
jaxphys solver (same update equations, same boundary handling, same
recording convention) with straightforward vectorized NumPy and a Python
time loop. They serve two purposes in the benchmark suite:

* a performance baseline ("what you would write without JAX"), and
* an independent check of the JAX results (agreement is asserted).

Analytic derivatives replace JAX autodiff where the solver differentiates a
user-supplied function (Hamiltonian / Lagrangian systems). The SPH reference
finds neighbours with SciPy's compiled ``cKDTree`` and the FDFD reference
uses SciPy's sparse direct solver, i.e. the usual tools one would reach for.
"""

from __future__ import annotations

import numpy as np

# Physical constants, as defined in jaxphys.em
C0 = 299792458.0
MU0 = 4.0e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = MU0 * C0
K_COULOMB = 8.9875517873681764e9


# ---------------------------------------------------------------------------
# Classical mechanics
# ---------------------------------------------------------------------------


def oscillator_leapfrog(
    q0: float, p0: float, k: float, dt: float, n_steps: int, save_every: int
) -> dict[str, np.ndarray]:
    """Leapfrog for H = p^2/2 + k q^2/2 (jaxphys ``leapfrog`` ordering)."""
    q, p = float(q0), float(p0)
    qs, ps = [q], [p]
    for i in range(1, n_steps + 1):
        p_half = p + 0.5 * dt * (-k * q)
        q = q + dt * p_half
        p = p_half + 0.5 * dt * (-k * q)
        if i % save_every == 0:
            qs.append(q)
            ps.append(p)
    q_arr, p_arr = np.array(qs), np.array(ps)
    return {"q": q_arr, "p": p_arr, "energy": 0.5 * p_arr**2 + 0.5 * k * q_arr**2}


def double_pendulum_rk4(
    q0: tuple[float, float],
    w0: tuple[float, float],
    g: float,
    dt: float,
    n_steps: int,
    save_every: int,
) -> dict[str, np.ndarray]:
    """RK4 for the unit-mass, unit-length double pendulum.

    Lagrangian: T = w1^2/2 + (w1^2 + w2^2 + 2 w1 w2 cos(t1 - t2))/2,
    V = -2 g cos t1 - g cos t2. Equations of motion solved with the 2x2
    mass matrix M = [[2, cos d], [cos d, 1]], d = t1 - t2.
    """

    def accel(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        d = q[0] - q[1]
        c, s = np.cos(d), np.sin(d)
        rhs = np.array(
            [
                -(w[1] ** 2) * s - 2.0 * g * np.sin(q[0]),
                w[0] ** 2 * s - g * np.sin(q[1]),
            ]
        )
        m = np.array([[2.0, c], [c, 1.0]])
        return np.linalg.solve(m, rhs)

    q = np.array(q0, dtype=float)
    w = np.array(w0, dtype=float)
    qs, ws = [q.copy()], [w.copy()]
    for i in range(1, n_steps + 1):
        k1q, k1w = w, accel(q, w)
        k2q, k2w = w + 0.5 * dt * k1w, accel(q + 0.5 * dt * k1q, w + 0.5 * dt * k1w)
        k3q, k3w = w + 0.5 * dt * k2w, accel(q + 0.5 * dt * k2q, w + 0.5 * dt * k2w)
        k4q, k4w = w + dt * k3w, accel(q + dt * k3q, w + dt * k3w)
        q = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
        w = w + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        if i % save_every == 0:
            qs.append(q.copy())
            ws.append(w.copy())
    return {"q": np.array(qs), "p": np.array(ws)}


def nbody_accel(pos: np.ndarray, m: np.ndarray, G: float, eps: float) -> np.ndarray:
    dr = pos[None, :, :] - pos[:, None, :]
    inv = (np.sum(dr**2, axis=-1) + eps**2) ** -1.5
    np.fill_diagonal(inv, 0.0)
    return G * np.einsum("j,ijk,ij->ik", m, dr, inv)


def nbody_verlet(
    pos: np.ndarray,
    vel: np.ndarray,
    m: np.ndarray,
    G: float,
    eps: float,
    dt: float,
    n_steps: int,
    save_every: int,
) -> dict[str, np.ndarray]:
    pos, vel = pos.copy(), vel.copy()
    acc = nbody_accel(pos, m, G, eps)
    ps, vs = [pos.copy()], [vel.copy()]
    for i in range(1, n_steps + 1):
        pos = pos + vel * dt + 0.5 * acc * dt**2
        acc_new = nbody_accel(pos, m, G, eps)
        vel = vel + 0.5 * (acc + acc_new) * dt
        acc = acc_new
        if i % save_every == 0:
            ps.append(pos.copy())
            vs.append(vel.copy())
    return {"positions": np.array(ps), "velocities": np.array(vs)}


def rigid_body_rk4(
    inertia: np.ndarray, omega0: np.ndarray, dt: float, n_steps: int
) -> dict[str, np.ndarray]:
    """Torque-free Euler equations + quaternion kinematics, RK4 (jaxphys form)."""
    I1, I2, I3 = inertia

    def dw(w: np.ndarray) -> np.ndarray:
        return np.array(
            [
                (I2 - I3) * w[1] * w[2] / I1,
                (I3 - I1) * w[2] * w[0] / I2,
                (I1 - I2) * w[0] * w[1] / I3,
            ]
        )

    def dq(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        a, x, y, z = q
        wx, wy, wz = w
        return 0.5 * np.array(
            [
                -x * wx - y * wy - z * wz,
                a * wx + y * wz - z * wy,
                a * wy + z * wx - x * wz,
                a * wz + x * wy - y * wx,
            ]
        )

    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.asarray(omega0, dtype=float)
    qs, wsave = [q.copy()], [w.copy()]
    for _ in range(n_steps):
        k1o = dw(w)
        k2o = dw(w + 0.5 * dt * k1o)
        k3o = dw(w + 0.5 * dt * k2o)
        k4o = dw(w + dt * k3o)
        w_new = w + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        k1q = dq(q, w)
        k2q = dq(q + 0.5 * dt * k1q, w + 0.5 * dt * k1o)
        k3q = dq(q + 0.5 * dt * k2q, w + 0.5 * dt * k2o)
        k4q = dq(q + dt * k3q, w + dt * k3o)
        q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
        q = q / np.linalg.norm(q)
        w = w_new
        qs.append(q.copy())
        wsave.append(w.copy())
    return {"q": np.array(qs), "p": np.array(wsave)}


# ---------------------------------------------------------------------------
# Electromagnetism
# ---------------------------------------------------------------------------


def charges_boris(
    pos: np.ndarray,
    vel: np.ndarray,
    q: np.ndarray,
    m: np.ndarray,
    B: np.ndarray,
    eps: float,
    dt: float,
    n_steps: int,
    save_every: int,
) -> dict[str, np.ndarray]:
    """Coulomb + uniform B: kick-drift-kick with Boris kicks (jaxphys form)."""
    qm = (q / m)[:, None]

    def kick(p: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
        dr = p[None, :, :] - p[:, None, :]
        inv = (np.sum(dr**2, axis=-1) + eps**2) ** -1.5
        np.fill_diagonal(inv, 0.0)
        e = -K_COULOMB * np.einsum("j,ijk,ij->ik", q, dr, inv)
        v_minus = v + qm * e * (0.5 * h)
        t = qm * np.broadcast_to(B, p.shape) * (0.5 * h)
        v_prime = v_minus + np.cross(v_minus, t)
        s = 2.0 * t / (1.0 + np.sum(t**2, axis=-1, keepdims=True))
        return v_minus + np.cross(v_prime, s) + qm * e * (0.5 * h)

    pos, vel = pos.copy(), vel.copy()
    ps, vs = [pos.copy()], [vel.copy()]
    for i in range(1, n_steps + 1):
        vel = kick(pos, vel, 0.5 * dt)
        pos = pos + dt * vel
        vel = kick(pos, vel, 0.5 * dt)
        if i % save_every == 0:
            ps.append(pos.copy())
            vs.append(vel.copy())
    return {"positions": np.array(ps), "velocities": np.array(vs)}


def _pml_loss(n: int, n_pml: int, dx: float, dt: float) -> tuple[np.ndarray, ...]:
    """Loss a = sigma dt / eps0 on nodes and half nodes (cubic grading)."""

    def loss(pos: np.ndarray) -> np.ndarray:
        if n_pml == 0:
            return np.zeros_like(pos)
        sigma_max = 0.8 * 4 / (ETA0 * dx)
        depth = np.clip(np.maximum(n_pml - pos, pos - (n - 1 - n_pml)) / n_pml, 0, 1)
        return sigma_max * depth**3 * dt / EPS0

    idx = np.arange(n, dtype=float)
    return loss(idx), loss(idx + 0.5)


def _pml_decay(n: int, n_pml: int, dx: float, dt: float) -> tuple[np.ndarray, ...]:
    """(exp(-a), (1 - exp(-a))/a) on nodes, then the same on half nodes."""

    def decay(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        safe = np.where(a > 0, a, 1.0)
        return np.exp(-a), np.where(a > 0, -np.expm1(-safe) / safe, 1.0)

    node, half = _pml_loss(n, n_pml, dx, dt)
    return (*decay(node), *decay(half))


def _fwd(a: np.ndarray, axis: int) -> np.ndarray:
    d = np.zeros_like(a)
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(0, -1)
    d[tuple(sl)] = np.diff(a, axis=axis)
    return d


def _bwd(a: np.ndarray, axis: int) -> np.ndarray:
    d = np.zeros_like(a)
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(1, None)
    d[tuple(sl)] = np.diff(a, axis=axis)
    return d


def fdtd2d_pml(
    nx: int,
    ny: int,
    dx: float,
    dt: float,
    n_steps: int,
    save_every: int,
    source_y: int,
    frequency: float,
    pml_layers: int = 10,
) -> dict[str, np.ndarray]:
    """2D TM Yee scheme with the split-field PML and a soft line source.

    Snapshots after steps 1, 1 + save_every, ... (jaxphys convention).
    """
    eax, ebx, hax, hbx = (a[:, None] for a in _pml_decay(nx, pml_layers, dx, dt))
    eay, eby, hay, hby = (a[None, :] for a in _pml_decay(ny, pml_layers, dx, dt))
    hc, ec = dt / (MU0 * dx), dt / (EPS0 * dx)
    omega = 2.0 * np.pi * frequency
    edge = np.zeros((nx, ny), bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    ezx, ezy, hx, hy = (np.zeros((nx, ny)) for _ in range(4))
    saved = []
    for i in range(n_steps):
        ez = ezx + ezy
        hx = hay * hx - hc * hby * _fwd(ez, 1)
        hy = hax * hy + hc * hbx * _fwd(ez, 0)
        ezx = eax * ezx + ec * ebx * _bwd(hy, 0)
        ezy = eay * ezy - ec * eby * _bwd(hx, 1)
        ezx[:, source_y] += np.sin(omega * i * dt)
        ezx[edge] = 0.0
        ezy[edge] = 0.0
        if i % save_every == 0:
            saved.append(ezx + ezy)
    return {"ez": np.array(saved)}


def fdtd3d_pml(
    n: int,
    dx: float,
    dt: float,
    n_steps: int,
    save_every: int,
    source: tuple[int, int, int],
    frequency: float,
    pml_layers: int = 8,
) -> dict[str, np.ndarray]:
    """3D Yee scheme, split-field PML, z-polarized soft point source.

    Snapshots of Ez after steps 1, 1 + save_every, ... (jaxphys convention).
    """
    dec = _pml_decay(n, pml_layers, dx, dt)
    shapes = [(n, 1, 1), (1, n, 1), (1, 1, n)]
    e_dec = [(dec[0].reshape(s), dec[1].reshape(s)) for s in shapes]
    h_dec = [(dec[2].reshape(s), dec[3].reshape(s)) for s in shapes]
    hc, ec = dt / (MU0 * dx), dt / (EPS0 * dx)

    def part(
        old: np.ndarray,
        d: tuple[np.ndarray, np.ndarray],
        coef: float,
        drive: np.ndarray,
    ) -> np.ndarray:
        return d[0] * old + coef * d[1] * drive

    edge = np.zeros((n, n, n), bool)
    edge[[0, -1]] = edge[:, [0, -1]] = edge[:, :, [0, -1]] = True
    f = [np.zeros((n, n, n)) for _ in range(12)]
    saved = []
    for step in range(n_steps):
        exy, exz, eyz, eyx, ezx, ezy, hxy, hxz, hyz, hyx, hzx, hzy = f
        ex, ey, ez = exy + exz, eyz + eyx, ezx + ezy
        hxy = part(hxy, h_dec[1], -hc, _fwd(ez, 1))
        hxz = part(hxz, h_dec[2], hc, _fwd(ey, 2))
        hyz = part(hyz, h_dec[2], -hc, _fwd(ex, 2))
        hyx = part(hyx, h_dec[0], hc, _fwd(ez, 0))
        hzx = part(hzx, h_dec[0], -hc, _fwd(ey, 0))
        hzy = part(hzy, h_dec[1], hc, _fwd(ex, 1))
        hx, hy, hz = hxy + hxz, hyz + hyx, hzx + hzy
        exy = part(exy, e_dec[1], ec, _bwd(hz, 1))
        exz = part(exz, e_dec[2], -ec, _bwd(hy, 2))
        eyz = part(eyz, e_dec[2], ec, _bwd(hx, 2))
        eyx = part(eyx, e_dec[0], -ec, _bwd(hz, 0))
        ezx = part(ezx, e_dec[0], ec, _bwd(hy, 0))
        ezy = part(ezy, e_dec[1], -ec, _bwd(hx, 1))
        ezx[source] += np.sin(2 * np.pi * frequency * step * dt)
        e = [np.where(edge, 0.0, a) for a in (exy, exz, eyz, eyx, ezx, ezy)]
        f = [*e, hxy, hxz, hyz, hyx, hzx, hzy]
        if step % save_every == 0:
            saved.append(f[4] + f[5])
    return {"ez": np.array(saved)}


def fdfd_tm(
    eps_r: np.ndarray,
    source: np.ndarray,
    frequency: float,
    dx: float,
    pml_layers: int,
) -> dict[str, np.ndarray]:
    """Stretched-coordinate PML 5-point TM system, SciPy sparse direct solve."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    nx, ny = eps_r.shape
    omega = 2.0 * np.pi * frequency

    def stretch(n: int) -> tuple[np.ndarray, np.ndarray]:
        # With dt = 1 the PML loss is sigma / eps0.
        node, half = _pml_loss(n, pml_layers, dx, 1.0)
        return 1.0 + 1j * node / omega, 1.0 + 1j * half / omega

    sx_e, sx_h = stretch(nx)
    sy_e, sy_h = stretch(ny)
    inv_dx2 = 1.0 / dx**2
    x_lo = inv_dx2 / (sx_e * np.concatenate([sx_h[:1], sx_h[:-1]]))
    x_hi = inv_dx2 / (sx_e * sx_h)
    y_lo = inv_dx2 / (sy_e * np.concatenate([sy_h[:1], sy_h[:-1]]))
    y_hi = inv_dx2 / (sy_e * sy_h)
    k0_sq = omega**2 * MU0 * EPS0
    center = k0_sq * eps_r - (x_lo[:, None] + x_hi[:, None] + y_lo + y_hi)

    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [center.ravel()]
    for coef, sl_row, sl_col in (
        (np.broadcast_to(x_lo[1:, None], (nx - 1, ny)), idx[1:], idx[:-1]),
        (np.broadcast_to(x_hi[:-1, None], (nx - 1, ny)), idx[:-1], idx[1:]),
        (np.broadcast_to(y_lo[None, 1:], (nx, ny - 1)), idx[:, 1:], idx[:, :-1]),
        (np.broadcast_to(y_hi[None, :-1], (nx, ny - 1)), idx[:, :-1], idx[:, 1:]),
    ):
        rows.append(sl_row.ravel())
        cols.append(sl_col.ravel())
        vals.append(coef.ravel())
    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nx * ny, nx * ny),
    )
    rhs = (-1j * omega * MU0 * source).ravel()
    return {"ez": spsolve(A, rhs).reshape(nx, ny)}


# ---------------------------------------------------------------------------
# Quantum mechanics
# ---------------------------------------------------------------------------


def split_operator(
    psi0: np.ndarray,
    V: np.ndarray,
    k2: np.ndarray,
    dt: float,
    n_steps: int,
    save_every: int,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> dict[str, np.ndarray]:
    """Strang split-step Fourier propagation (V/2, T, V/2), any dimension."""
    ev = np.exp(-1j * V * dt / (2.0 * hbar))
    et = np.exp(-1j * hbar * k2 * dt / (2.0 * mass))
    psi = psi0.astype(np.complex128)
    saved = [psi.copy()]
    for i in range(1, n_steps + 1):
        psi = ev * np.fft.ifftn(et * np.fft.fftn(ev * psi))
        if i % save_every == 0:
            saved.append(psi.copy())
    return {"psi": np.array(saved)}


def lindblad_rk4(
    rho0: np.ndarray,
    H: np.ndarray,
    ops: list[np.ndarray],
    rates: list[float],
    dt: float,
    n_steps: int,
    save_every: int,
    hbar: float = 1.0,
) -> dict[str, np.ndarray]:
    def rhs(r: np.ndarray) -> np.ndarray:
        d = -1j / hbar * (H @ r - r @ H)
        for g, L in zip(rates, ops, strict=True):
            Ld = L.conj().T
            LdL = Ld @ L
            d = d + g * (L @ r @ Ld - 0.5 * (LdL @ r + r @ LdL))
        return d

    rho = rho0.astype(np.complex128)
    saved = [rho.copy()]
    for i in range(1, n_steps + 1):
        k1 = rhs(rho)
        k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2)
        k4 = rhs(rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho)
        if i % save_every == 0:
            saved.append(rho.copy())
    return {"rho": np.array(saved)}


def honeycomb_bands(t: float, a: float, k_points: np.ndarray) -> np.ndarray:
    """Graphene nearest-neighbour bands, dense 2x2 eigvalsh per k."""
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3.0) / 2.0])
    # Bonds from A (orbital 0) to B (orbital 1) in cells 0, -a1, -a2.
    f = t * (
        1.0 + np.exp(-1j * k_points @ a1) + np.exp(-1j * k_points @ a2)
    )  # shape (n_k,)
    H = np.zeros((len(k_points), 2, 2), complex)
    H[:, 0, 1] = f
    H[:, 1, 0] = f.conj()
    return np.linalg.eigvalsh(H)


# ---------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------

_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = _CX * ux[..., None] + _CY * uy[..., None]
    usq = (ux**2 + uy**2)[..., None]
    return _W * rho[..., None] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)


def lbm_d2q9(
    nx: int,
    ny: int,
    tau: float,
    u_inlet: float,
    n_steps: int,
    save_every: int,
    obstacle: np.ndarray,
) -> dict[str, np.ndarray]:
    """D2Q9 BGK, periodic y, jaxphys boundaries.

    Solid nodes send back (unchanged, uncollided) the populations that
    streamed into them; Zou-He velocity inlet at x = 0, Zou-He pressure
    outlet (rho = 1) at x = nx - 1. Snapshots at steps 0, s, 2s, ...
    """
    omega = 1.0 / tau
    f = _feq(np.ones((nx, ny)), np.full((nx, ny), u_inlet), np.zeros((nx, ny)))
    f[obstacle] = _W
    fluid_in, fluid_out = ~obstacle[0], ~obstacle[-1]
    saved: dict[str, list[np.ndarray]] = {"rho": [], "ux": [], "uy": []}

    def record(f: np.ndarray) -> None:
        rho = f.sum(-1)
        saved["rho"].append(rho)
        saved["ux"].append(np.where(obstacle, 0.0, (f * _CX).sum(-1) / rho))
        saved["uy"].append(np.where(obstacle, 0.0, (f * _CY).sum(-1) / rho))

    record(f)
    for i in range(1, n_steps + 1):
        rho = f.sum(-1)
        ux = (f * _CX).sum(-1) / rho
        uy = (f * _CY).sum(-1) / rho
        post = f - omega * (f - _feq(rho, ux, uy))
        post[obstacle] = f[obstacle][:, _OPP]
        for d in range(9):
            f[..., d] = np.roll(post[..., d], (_CX[d], _CY[d]), axis=(0, 1))
        c = f[0].copy()
        r_in = (c[:, 0] + c[:, 2] + c[:, 4] + 2.0 * (c[:, 3] + c[:, 6] + c[:, 7])) / (
            1.0 - u_inlet
        )
        shear = 0.5 * (c[:, 2] - c[:, 4])
        c[:, 1] = c[:, 3] + (2.0 / 3.0) * r_in * u_inlet
        c[:, 5] = c[:, 7] - shear + (1.0 / 6.0) * r_in * u_inlet
        c[:, 8] = c[:, 6] + shear + (1.0 / 6.0) * r_in * u_inlet
        f[0, fluid_in] = c[fluid_in]
        c = f[-1].copy()
        u_out = -1.0 + (
            c[:, 0] + c[:, 2] + c[:, 4] + 2.0 * (c[:, 1] + c[:, 5] + c[:, 8])
        )
        shear = 0.5 * (c[:, 2] - c[:, 4])
        c[:, 3] = c[:, 1] - (2.0 / 3.0) * u_out
        c[:, 7] = c[:, 5] + shear - (1.0 / 6.0) * u_out
        c[:, 6] = c[:, 8] - shear - (1.0 / 6.0) * u_out
        f[-1, fluid_out] = c[fluid_out]
        if i % save_every == 0:
            record(f)
    return {k: np.array(v) for k, v in saved.items()}


def vorticity_streamfunction(
    n: int,
    nu: float,
    dt: float,
    lid: float,
    poisson_iters: int,
    n_steps: int,
    save_every: int,
    dx: float = 1.0,
) -> dict[str, np.ndarray]:
    """Lid-driven cavity, jaxphys' explicit vorticity-streamfunction scheme.

    The Jacobi iteration for psi is warm-started from the previous step and
    run after the vorticity update; snapshots at steps 0, s, 2s, ...
    """
    w = np.zeros((n, n))

    def poisson(psi: np.ndarray, w: np.ndarray) -> np.ndarray:
        for _ in range(poisson_iters):
            new = np.zeros_like(psi)
            new[1:-1, 1:-1] = (
                0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2])
                + 0.25 * dx**2 * w[1:-1, 1:-1]
            )
            psi = new
        return psi

    def velocity(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
        u[:, -1] = lid
        return u, v

    psi = poisson(np.zeros((n, n)), w)
    saved = [velocity(psi)[0]]
    for i in range(1, n_steps + 1):
        u, v = velocity(psi)
        c = w[1:-1, 1:-1]
        dwdx = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2.0 * dx)
        dwdy = (w[1:-1, 2:] - w[1:-1, :-2]) / (2.0 * dx)
        lap = (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2]) / dx**2 - (
            4.0 * c / dx**2
        )
        w = w.copy()
        w[1:-1, 1:-1] = c + dt * (
            -u[1:-1, 1:-1] * dwdx - v[1:-1, 1:-1] * dwdy + nu * lap
        )
        w[:, -1] = -2.0 * psi[:, -2] / dx**2 - 2.0 * lid / dx
        w[:, 0] = -2.0 * psi[:, 1] / dx**2
        w[0, :] = -2.0 * psi[1, :] / dx**2
        w[-1, :] = -2.0 * psi[-2, :] / dx**2
        psi = poisson(psi, w)
        if i % save_every == 0:
            saved.append(velocity(psi)[0])
    return {"ux": np.array(saved)}


def sph_periodic(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: float,
    h: float,
    box: tuple[float, float],
    rest_density: float,
    sound_speed: float,
    gamma: float,
    alpha: float,
    dt: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Weakly compressible SPH, kick-drift-kick, cKDTree neighbour search."""
    from scipy.spatial import cKDTree

    L = np.asarray(box)
    sigma = 10.0 / (7.0 * np.pi * h**2)
    B = rest_density * sound_speed**2 / gamma

    def acceleration(p: np.ndarray, v: np.ndarray) -> np.ndarray:
        pairs = cKDTree(p, boxsize=L).query_pairs(2 * h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        dr = p[i] - p[j]
        dr -= L * np.round(dr / L)
        r = np.linalg.norm(dr, axis=1)
        q = r / h
        w = sigma * np.where(q < 1, 1 - 1.5 * q**2 + 0.75 * q**3, 0.25 * (2 - q) ** 3)
        dw = sigma / h * np.where(q < 1, -3 * q + 2.25 * q**2, -0.75 * (2 - q) ** 2)
        n = len(p)
        rho = mass * (sigma + np.bincount(i, w, n) + np.bincount(j, w, n))
        pr = B * ((rho / rest_density) ** gamma - 1)
        vr = np.sum((v[i] - v[j]) * dr, axis=1)
        mu = h * vr / (r**2 + 0.01 * h**2)
        visc = np.where(
            vr < 0, -alpha * sound_speed * mu / (0.5 * (rho[i] + rho[j])), 0.0
        )
        f = (-mass * (pr[i] / rho[i] ** 2 + pr[j] / rho[j] ** 2 + visc) * dw / r)[
            :, None
        ] * dr
        return np.stack(
            [np.bincount(i, f[:, d], n) - np.bincount(j, f[:, d], n) for d in (0, 1)], 1
        )

    a = acceleration(pos, vel)
    for _ in range(n_steps):
        v_half = vel + 0.5 * dt * a
        pos = np.mod(pos + dt * v_half, L)
        a = acceleration(pos, v_half)
        vel = v_half + 0.5 * dt * a
    return {"positions": pos, "velocities": vel}


def euler_muscl_hllc(
    rho: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    dt_dx: float,
    gamma: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """1D Euler: MUSCL (minmod) + HLLC + SSP-RK2, transmissive boundaries."""
    g = gamma

    def primitive(U: np.ndarray) -> np.ndarray:
        r = U[0]
        v = U[1] / r
        return np.stack([r, v, (g - 1.0) * (U[2] - 0.5 * r * v**2)])

    def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def hllc(WL: np.ndarray, WR: np.ndarray) -> np.ndarray:
        rL, uL, pL = WL
        rR, uR, pR = WR
        cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
        EL = pL / (g - 1.0) + 0.5 * rL * uL**2
        ER = pR / (g - 1.0) + 0.5 * rR * uR**2
        SL = np.minimum(uL - cL, uR - cR)
        SR = np.maximum(uL + cL, uR + cR)
        Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (
            rL * (SL - uL) - rR * (SR - uR)
        )

        def flux(r: np.ndarray, v: np.ndarray, pr: np.ndarray, E: np.ndarray):  # type: ignore[no-untyped-def]
            return np.stack([r * v, r * v**2 + pr, (E + pr) * v])

        def star(r, v, pr, E, S):  # type: ignore[no-untyped-def]
            factor = r * (S - v) / (S - Ss)
            energy = E / r + (Ss - v) * (Ss + pr / (r * (S - v)))
            return factor * np.stack([np.ones_like(r), Ss, energy])

        UL = np.stack([rL, rL * uL, EL])
        UR = np.stack([rR, rR * uR, ER])
        FL, FR = flux(rL, uL, pL, EL), flux(rR, uR, pR, ER)
        FLs = FL + SL * (star(rL, uL, pL, EL, SL) - UL)
        FRs = FR + SR * (star(rR, uR, pR, ER, SR) - UR)
        return np.where(SL >= 0, FL, np.where(Ss >= 0, FLs, np.where(SR > 0, FRs, FR)))

    def rhs(U: np.ndarray) -> np.ndarray:
        W = primitive(U)
        W = np.concatenate([W[:, :1].repeat(2, 1), W, W[:, -1:].repeat(2, 1)], axis=1)
        slope = minmod(W[:, 1:-1] - W[:, :-2], W[:, 2:] - W[:, 1:-1])
        F = hllc(W[:, 1:-2] + 0.5 * slope[:, :-1], W[:, 2:-1] - 0.5 * slope[:, 1:])
        return -(F[:, 1:] - F[:, :-1])

    U = np.stack([rho, rho * u, p / (g - 1.0) + 0.5 * rho * u**2])
    for _ in range(n_steps):
        U1 = U + dt_dx * rhs(U)
        U = 0.5 * (U + U1 + dt_dx * rhs(U1))
    W = primitive(U)
    return {"rho": W[0], "u": W[1], "p": W[2]}


# ---------------------------------------------------------------------------
# Statistical mechanics (different RNG streams: compared statistically)
# ---------------------------------------------------------------------------


def ising_metropolis(
    L: int, T: float, n_sweeps: int, n_warmup: int, seed: int
) -> dict[str, np.ndarray]:
    """Random-site single-spin-flip Metropolis.

    jaxphys uses checkerboard sweeps on even lattices; both sample the same
    Boltzmann distribution, so the comparison is statistical.
    """
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1]), size=(L, L))
    beta = 1.0 / T
    energies, mags = [], []
    for sweep in range(n_warmup + n_sweeps):
        ix = rng.integers(0, L, L * L)
        iy = rng.integers(0, L, L * L)
        u = rng.random(L * L)
        for x, y, r in zip(ix, iy, u, strict=True):
            nn = s[(x + 1) % L, y] + s[x - 1, y] + s[x, (y + 1) % L] + s[x, y - 1]
            dE = 2.0 * s[x, y] * nn
            if dE < 0 or r < np.exp(-beta * dE):
                s[x, y] = -s[x, y]
        if sweep >= n_warmup:
            e = -np.sum(s * np.roll(s, 1, 0) + s * np.roll(s, 1, 1)) / (L * L)
            energies.append(e)
            mags.append(abs(s.mean()))
    return {"energy": np.array(energies), "magnetization": np.array(mags)}


def ising_wolff(
    L: int, T: float, n_updates: int, n_warmup: int, seed: int
) -> dict[str, np.ndarray]:
    """Textbook Wolff algorithm: every bond from a cluster site is tried once."""
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1]), size=(L, L))
    p_add = 1.0 - np.exp(-2.0 / T)
    energies, mags = [], []
    for it in range(n_warmup + n_updates):
        x0, y0 = rng.integers(0, L, 2)
        spin = s[x0, y0]
        in_cluster = np.zeros((L, L), bool)
        in_cluster[x0, y0] = True
        stack = [(x0, y0)]
        while stack:
            x, y = stack.pop()
            for nx_, ny_ in (
                ((x + 1) % L, y),
                ((x - 1) % L, y),
                (x, (y + 1) % L),
                (x, (y - 1) % L),
            ):
                if (
                    not in_cluster[nx_, ny_]
                    and s[nx_, ny_] == spin
                    and rng.random() < p_add
                ):
                    in_cluster[nx_, ny_] = True
                    stack.append((nx_, ny_))
        s[in_cluster] *= -1
        if it >= n_warmup:
            e = -np.sum(s * np.roll(s, 1, 0) + s * np.roll(s, 1, 1)) / (L * L)
            energies.append(e)
            mags.append(abs(s.mean()))
    return {"energy": np.array(energies), "magnetization": np.array(mags)}
