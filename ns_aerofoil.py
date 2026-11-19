#!/usr/bin/env python3
"""
ns_aerofoil.py
2D incompressible Navier-Stokes flow past an aerofoil (or a cylinder).

How it works (the doc explains every step):
  1. Staggered "MAC" grid: u on vertical cell faces, v on horizontal faces,
     pressure at cell centres.
  2. Each time step: move the velocity forward with advection and viscosity,
     hold the fluid still inside the body, then solve a pressure Poisson
     equation that removes any divergence (the projection method).
  3. Third-order upwind advection, central-difference viscosity,
     three-stage Runge-Kutta (SSP-RK3) in time, time step set by the CFL number.
  4. The body is a mask on the square grid (immersed boundary). The momentum
     the mask removes from the fluid each step is the force on the body,
     which gives lift and drag.
  5. The pressure Poisson equation is solved exactly with a 2D cosine transform.
  6. Simulated smoke: a dye released at the inlet at the heights of your incense
     sticks and carried by the flow, so it can sit next to your smoke photos.
  7. Flow angle at point P (0.5 chord ahead of and 0.3 chord above the rod),
     averaged over time: how steeply the air rises before it reaches the
     aerofoil (upwash). It grows with lift, and you read the same angle off
     your smoke photos.

Units: chord (or cylinder diameter) = 1, free-stream speed = 1, density = 1.
So viscosity nu = 1/Re and time t counts chords travelled by the free stream.

Run it:
  python3 ns_aerofoil.py               use SETTINGS below
  python3 ns_aerofoil.py --quick       coarse grid, short run (about a minute): checks it works
  python3 ns_aerofoil.py --validate    cylinder at Re 100, compared with published values
  python3 ns_aerofoil.py --sweep       angle-of-attack sweep: CSV + lift-curve and flow-angle plots
  python3 ns_aerofoil.py --alpha 8 --re 2500 --mode free     (any setting from the command line)

Needs: python3 -m pip install numpy scipy matplotlib
"""

import argparse, csv, os, sys, time
import numpy as np
from scipy.fft import dctn, idctn
import matplotlib
matplotlib.use("Agg")                      # draw straight to files, no windows
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ------------------------------------------------------------------ SETTINGS
SETTINGS = dict(
    shape="NACA2412",        # "NACA2412", "NACA0012", any NACA 4-digit, or "cylinder"
    alpha_deg=6.0,           # angle of attack in degrees, nose up positive
    Re=1500,                 # Reynolds number on chord. Smoke tube, 30 mm chord: Re ~ 1,600 x speed in m/s
    mode="tunnel",           # "tunnel": walls tunnel_height chords apart, like your smoke chamber
                             # "free":   walls 6 chords apart, closer to open air
    tunnel_height=2.0,       # tube width / chord: 60 mm / 30 mm = 2.0
    smoke_lines=[-0.333, 0.0, 0.333],          # incense positions relative to the axle, in chords
                             # (30 mm chord: 10 mm to one side, level with, 10 mm to the other side)
    cells_per_chord=100,     # grid resolution; 64 is quick, 100 is the default, 150 checks the grid
    t_end=16.0,              # how long to simulate, in chords travelled (averages use the second half)
    cfl=0.5,                 # time-step safety number (keep 0.3 to 0.6)
    frame_every=0.25,        # save a picture every this much simulated time (0 = none)
    colour=False,            # False: black-and-white pictures that print well
    make_gif=True,           # stitch the pictures into an animation at the end
    out_dir="results",
)
SWEEP_ALPHAS = [0, 4, 8, 12, 16, 20]
TE_THICKNESS = 0.8/30        # trailing edge 0.8 mm thick on a 30 mm chord, the same as the printed parts
PROBE = (-0.5, 0.3)          # point P, in chords from the axle: 15 mm ahead, 9 mm to the upper side (30 mm chord)


# ------------------------------------------------------------------ GEOMETRY
def naca4(code, n=200, te=TE_THICKNESS):
    """NACA 4-digit outline, chord 1, as a closed loop of points (x, y)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))                         # points bunched at the nose and tail
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yt += x * max(0.0, te - 2*yt[-1]) / 2                # thicken the trailing edge like the print
    if m == 0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1 - 2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    px = np.concatenate([xu[::-1], xl[1:]])
    py = np.concatenate([yu[::-1], yl[1:]])
    yq = 0.0 if m == 0 else (m/p**2*(2*p*0.25 - 0.25**2) if 0.25 < p else m/(1-p)**2*((1-2*p) + 2*p*0.25 - 0.25**2))
    return px, py, (0.25, yq)                            # outline and the pivot (rod) point


def body_outline(shape, alpha_deg):
    """Outline of the body in flow coordinates: pivot at the origin, flow along +x."""
    if shape.lower() == "cylinder":
        th = np.linspace(0, 2*np.pi, 400, endpoint=False)
        return 0.5*np.cos(th), 0.5*np.sin(th)
    px, py, (xq, yq) = naca4(shape.upper().replace("NACA", ""))
    a = -np.radians(alpha_deg)                           # nose up = clockwise when flow goes left to right
    x, y = px - xq, py - yq
    return x*np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a)


def solid_fraction(X, Y, shape, outline, h, sub=4):
    """Fraction of each little square (side h) around the points (X, Y) that lies inside the body."""
    chi = np.zeros(X.shape)
    bx, by = outline
    box = (X > bx.min() - h) & (X < bx.max() + h) & (Y > by.min() - h) & (Y < by.max() + h)
    xs, ys = X[box], Y[box]
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * h
    ox, oy = np.meshgrid(off, off, indexing="ij")
    px = (xs[:, None] + ox.ravel()[None, :]).ravel()
    py = (ys[:, None] + oy.ravel()[None, :]).ravel()
    if shape.lower() == "cylinder":
        inside = px**2 + py**2 <= 0.25
    else:
        inside = Path(np.c_[bx, by]).contains_points(np.c_[px, py])
    chi[box] = inside.reshape(-1, sub*sub).mean(axis=1)
    return chi


# ------------------------------------------------------------------ SOLVER
class Solver:
    def __init__(self, s):
        self.s = s
        cyl = s["shape"].lower() == "cylinder"
        if cyl:                                          # validation box: diameter 1, walls 20 diameters apart
            x0, x1, y0, y1 = -5.0, 15.0, -10.0, 10.0
        elif s["mode"] == "tunnel":                      # inlet 1.75 chords ahead of the rod, as in the chamber
            hh = 0.5*s["tunnel_height"]
            x0, x1, y0, y1 = -1.75, 4.25, -hh, hh
        else:
            x0, x1, y0, y1 = -2.5, 5.5, -3.0, 3.0
        h = 1.0 / s["cells_per_chord"]
        self.nx, self.ny = int(round((x1 - x0) / h)), int(round((y1 - y0) / h))
        self.h, self.x0, self.y0 = h, x0, y0
        self.nu = 1.0 / s["Re"]
        nx, ny = self.nx, self.ny

        # where each unknown lives on the staggered grid
        xf, yf = x0 + h*np.arange(nx + 1), y0 + h*np.arange(ny + 1)        # faces
        xc, yc = x0 + h*(np.arange(nx) + 0.5), y0 + h*(np.arange(ny) + 0.5)  # centres
        self.xc, self.yc, self.xf, self.yf = xc, yc, xf, yf

        self.outline = body_outline(s["shape"], s["alpha_deg"])
        Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
        Xv, Yv = np.meshgrid(xc, yf, indexing="ij")
        self.chi_u = solid_fraction(Xu, Yu, s["shape"], self.outline, h)
        self.chi_v = solid_fraction(Xv, Yv, s["shape"], self.outline, h)

        # start: uniform flow, like switching the fan on
        self.u = np.ones((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        if cyl:                                          # a nudge so vortex shedding starts sooner
            self.v += 0.5 * np.exp(-((Xv - 1.0)**2 + (Yv - 0.5)**2) / 0.5)
        self.p = np.zeros((nx, ny))

        # simulated smoke: dye entering at the inlet in thin bands at the incense heights
        self.c = np.zeros((nx, ny))
        band = 0.03 if not cyl else 0.06
        self.c_in = np.zeros(ny)
        for yl in s["smoke_lines"]:
            self.c_in[np.abs(yc - yl) < band] = 1.0

        # time-averaged velocity (for the mean streamlines and the flow angle at P)
        self.u_sum, self.v_sum, self.t_sum = np.zeros_like(self.u), np.zeros_like(self.v), 0.0
        self.t_avg = np.inf

        # eigenvalues of the pressure Laplacian for the cosine-transform solver
        kx = 2*np.cos(np.pi*np.arange(nx)/nx) - 2
        ky = 2*np.cos(np.pi*np.arange(ny)/ny) - 2
        lam = (kx[:, None] + ky[None, :]) / h**2
        lam[0, 0] = 1.0
        self.inv_lam = 1.0 / lam
        self.inv_lam[0, 0] = 0.0

        self.t = 0.0
        self.u, self.v, _, _, _ = self.stage(self.u, self.v)   # body in place before the first step

    # ---- ghost cells: two extra layers round each array, filled from the boundary conditions
    def pad_u(self, u):
        P = np.empty((u.shape[0] + 4, u.shape[1] + 4))
        P[2:-2, 2:-2] = u
        P[:2, 2:-2] = 1.0                                # inflow: free stream
        P[-2:, 2:-2] = u[-1]                             # outflow: zero gradient
        P[:, 1], P[:, 0] = P[:, 2], P[:, 3]              # bottom wall: free slip (mirror)
        P[:, -2], P[:, -1] = P[:, -3], P[:, -4]          # top wall: free slip (mirror)
        return P

    def pad_v(self, v):
        P = np.empty((v.shape[0] + 4, v.shape[1] + 4))
        P[2:-2, 2:-2] = v
        P[1, 2:-2], P[0, 2:-2] = -v[0], -v[1]            # inflow: v = 0 on the boundary (odd mirror)
        P[-2:, 2:-2] = v[-1]                             # outflow: zero gradient
        P[:, 1], P[:, 0] = -P[:, 3], -P[:, 4]            # walls: v = 0 on the wall itself (odd mirror)
        P[:, -2], P[:, -1] = -P[:, -4], -P[:, -5]
        return P

    @staticmethod
    def upwind3(fm2, fm1, f0, fp1, fp2, vel, h):
        """vel * df/dx with third-order upwinding: central part + a small built-in damping of wiggles.
        Same as  vel*(-fp2 + 8(fp1 - fm1) + fm2)/12h + |vel|*(fp2 - 4fp1 + 6f0 - 4fm1 + fm2)/12h,
        written with in-place operations because this line is where most of the run time goes."""
        c = fp1 - fm1; c *= 8; c += fm2; c -= fp2           # central part (times 12h)
        s = fp1 + fm1; s *= -4
        d = fp2 + fm2; d += s                               # damping part (times 12h)
        np.multiply(f0, 6, out=s); d += s
        c *= vel
        np.abs(vel, out=s); d *= s
        c += d; c *= 1.0/(12*h)
        return c

    def rhs(self, u, v):
        """Rate of change of u and v from advection and viscosity (pressure comes later)."""
        h, nu, nx, ny = self.h, self.nu, self.nx, self.ny
        U, V = self.pad_u(u), self.pad_v(v)
        ru, rv = np.zeros_like(u), np.zeros_like(v)

        # u equation on the interior vertical faces i = 1 .. nx-1
        sx = [U[3 + k:nx + 2 + k, 2:-2] for k in (-2, -1, 0, 1, 2)]
        sy = [U[3:nx + 2, 2 + k:ny + 2 + k] for k in (-2, -1, 0, 1, 2)]
        uc = sx[2]
        v_at_u = 0.25*(v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        adv = self.upwind3(*sx, uc, h) + self.upwind3(*sy, v_at_u, h)
        lap = (sx[1] + sx[3] + sy[1] + sy[3] - 4*uc) / h**2
        ru[1:nx] = -adv + nu*lap
        ru[nx] = -(u[nx] - u[nx - 1]) / h                # outflow: disturbances carried out at speed 1

        # v equation on the interior horizontal faces j = 1 .. ny-1
        sx = [V[2 + k:nx + 2 + k, 3:ny + 2] for k in (-2, -1, 0, 1, 2)]
        sy = [V[2:nx + 2, 3 + k:ny + 2 + k] for k in (-2, -1, 0, 1, 2)]
        vc = sy[2]
        u_at_v = 0.25*(u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
        adv = self.upwind3(*sx, u_at_v, h) + self.upwind3(*sy, vc, h)
        lap = (sx[1] + sx[3] + sy[1] + sy[3] - 4*vc) / h**2
        rv[:, 1:ny] = -adv + nu*lap
        return ru, rv

    def project(self, u, v):
        """Make the velocity divergence-free: solve lap(phi) = div(u), then u -= grad(phi)."""
        h = self.h
        u[0] = 1.0                                       # inflow
        v[:, 0] = 0.0; v[:, -1] = 0.0                    # walls
        u[-1] += (u[0].sum() - u[-1].sum()) / self.ny    # outflow carries exactly what comes in
        div = (u[1:] - u[:-1] + v[:, 1:] - v[:, :-1]) / h
        phi = idctn(dctn(div, type=2, norm="ortho") * self.inv_lam, type=2, norm="ortho")
        u[1:-1] -= (phi[1:] - phi[:-1]) / h
        v[:, 1:-1] -= (phi[:, 1:] - phi[:, :-1]) / h
        return u, v, phi

    def stage(self, u, v):
        """Hold the fluid still inside the body, then project. Returns what the body removed."""
        Ru, Rv = self.chi_u * u, self.chi_v * v
        u -= Ru; v -= Rv
        u, v, phi = self.project(u, v)
        return u, v, phi, Ru.sum(), Rv.sum()

    def pad_c(self, c):
        P = np.empty((c.shape[0] + 4, c.shape[1] + 4))
        P[2:-2, 2:-2] = c
        P[:2, 2:-2] = self.c_in                          # inlet: smoke bands
        P[-2:, 2:-2] = c[-1]                             # outlet: zero gradient
        P[:, 1], P[:, 0] = P[:, 2], P[:, 3]              # walls: zero gradient
        P[:, -2], P[:, -1] = P[:, -3], P[:, -4]
        return P

    def rhs_c(self, c, uc, vc):
        nx, ny, h = self.nx, self.ny, self.h
        C = self.pad_c(c)
        sx = [C[2 + k:nx + 2 + k, 2:-2] for k in (-2, -1, 0, 1, 2)]
        sy = [C[2:-2, 2 + k:ny + 2 + k] for k in (-2, -1, 0, 1, 2)]
        return -(self.upwind3(*sx, uc, h) + self.upwind3(*sy, vc, h))

    def step(self):
        h = self.h
        umax = np.abs(self.u).max() + np.abs(self.v).max() + 1e-12
        dt = min(self.s["cfl"]*h/umax, 0.25*h*h/self.nu)
        u0, v0 = self.u, self.v
        # SSP-RK3 (Shu-Osher form), with the body and the projection applied at every stage
        ru, rv = self.rhs(u0, v0)
        u1, v1, _, a1, b1 = self.stage(u0 + dt*ru, v0 + dt*rv)
        ru, rv = self.rhs(u1, v1)
        u2, v2, _, a2, b2 = self.stage(0.75*u0 + 0.25*(u1 + dt*ru), 0.75*v0 + 0.25*(v1 + dt*rv))
        ru, rv = self.rhs(u2, v2)
        u3, v3, phi, a3, b3 = self.stage(u0/3 + (2/3)*(u2 + dt*ru), v0/3 + (2/3)*(v2 + dt*rv))
        # smoke: carried by the velocity from the start of the step (SSP-RK3 again)
        uc, vc = 0.5*(u0[1:] + u0[:-1]), 0.5*(v0[:, 1:] + v0[:, :-1])
        c0 = self.c
        c1 = c0 + dt*self.rhs_c(c0, uc, vc)
        c2 = 0.75*c0 + 0.25*(c1 + dt*self.rhs_c(c1, uc, vc))
        self.c = np.clip(c0/3 + (2/3)*(c2 + dt*self.rhs_c(c2, uc, vc)), 0.0, 1.0)
        self.u, self.v = u3, v3
        self.p = phi / (2*dt/3)
        self.t += dt
        if self.t >= self.t_avg:                         # running time-average
            self.u_sum += dt*self.u; self.v_sum += dt*self.v; self.t_sum += dt
        # momentum removed per unit time = force on the body (per unit span); weights come from RK3
        Fx = (a1/6 + 2*a2/3 + a3) * h*h / dt
        Fy = (b1/6 + 2*b2/3 + b3) * h*h / dt
        return dt, 2*Fx, 2*Fy                             # drag and lift coefficients (rho = U = c = 1)

    # ---- fields for pictures
    def vorticity(self):
        h = self.h
        return (self.v[1:, 1:-1] - self.v[:-1, 1:-1]) / h - (self.u[1:-1, 1:] - self.u[1:-1, :-1]) / h

    def centre_velocity(self, mean=False):
        u = self.u_sum/self.t_sum if mean else self.u
        v = self.v_sum/self.t_sum if mean else self.v
        return 0.5*(u[1:] + u[:-1]), 0.5*(v[:, 1:] + v[:, :-1])

    def probe_angle(self):
        """Time-averaged flow angle at point P, in degrees (positive = flow heading up)."""
        if self.t_sum == 0:
            return float("nan")
        uc, vc = self.centre_velocity(mean=True)
        xq, yq = PROBE
        fi, fj = (xq - self.xc[0])/self.h, (yq - self.yc[0])/self.h
        i, j = int(fi), int(fj); a, b = fi - i, fj - j
        def bil(F):
            return (1-a)*(1-b)*F[i, j] + a*(1-b)*F[i+1, j] + (1-a)*b*F[i, j+1] + a*b*F[i+1, j+1]
        return float(np.degrees(np.arctan2(bil(vc), bil(uc))))


# ------------------------------------------------------------------ PICTURES
def draw_body(ax, outline):
    ax.fill(outline[0], outline[1], facecolor="white", edgecolor="black", hatch="////", lw=1.4, zorder=5)


def view_box(sv):
    if sv.s["shape"].lower() == "cylinder":
        return (-2, 10, -3, 3)
    return (sv.xf[0], sv.xf[-1], sv.yf[0], sv.yf[-1])


def save_vorticity(sv, path, title):
    w = sv.vorticity()
    vmax = 5.0 if sv.s["Re"] <= 200 else 15.0
    cmap = "RdBu_r" if sv.s["colour"] else "gray"
    x0, x1, y0, y1 = view_box(sv)
    fig, ax = plt.subplots(figsize=(10, 10*(y1 - y0)/(x1 - x0) + 0.8))
    ax.imshow(w.T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="bilinear",
              extent=(sv.xf[1], sv.xf[-2], sv.yf[1], sv.yf[-2]))
    draw_body(ax, sv.outline)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    unit = "diameter" if sv.s["shape"].lower() == "cylinder" else "chord"
    ax.set_xlabel(f"x / {unit}"); ax.set_ylabel(f"y / {unit}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_pressure(sv, path, title):
    uc, vc = sv.centre_velocity()
    cp = 2*(sv.p - sv.p[-1].mean())                      # pressure coefficient, outlet as reference
    x0, x1, y0, y1 = view_box(sv)
    fig, ax = plt.subplots(figsize=(10, 10*(y1 - y0)/(x1 - x0) + 0.8))
    im = ax.imshow(cp.T, origin="lower", cmap="RdBu_r" if sv.s["colour"] else "Greys_r",
                   vmin=-2, vmax=1, extent=(sv.xf[0], sv.xf[-1], sv.yf[0], sv.yf[-1]))
    ax.streamplot(sv.xc, sv.yc, uc.T, vc.T, color="black", linewidth=0.6, density=1.4, arrowstyle="-")
    draw_body(ax, sv.outline)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="pressure coefficient Cp", shrink=0.8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_smoke(sv, path, title):
    """Simulated incense smoke: white dye on black, like the photo against your black back wall."""
    x0, x1, y0, y1 = view_box(sv)
    fig, ax = plt.subplots(figsize=(10, 10*(y1 - y0)/(x1 - x0) + 0.8))
    ax.imshow(sv.c.T, origin="lower", cmap="gray", vmin=0, vmax=0.8, interpolation="bilinear",
              extent=(sv.xf[0], sv.xf[-1], sv.yf[0], sv.yf[-1]))
    draw_body(ax, sv.outline)
    if sv.s["shape"].lower() != "cylinder":
        ax.plot(*PROBE, "o", mfc="none", mec="white", ms=7, mew=1.2, zorder=6)
        ax.text(PROBE[0] + 0.05, PROBE[1] + 0.06, "P", color="white", fontsize=10, zorder=6)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_mean_streamlines(sv, path, title, angle):
    """Time-averaged streamlines; a separated region shows up as a closed loop above the aerofoil."""
    uc, vc = sv.centre_velocity(mean=True)
    x0, x1, y0, y1 = view_box(sv)
    fig, ax = plt.subplots(figsize=(10, 10*(y1 - y0)/(x1 - x0) + 0.8))
    ax.streamplot(sv.xc, sv.yc, uc.T, vc.T, color="black", linewidth=0.7, density=2.0, arrowsize=0.7)
    draw_body(ax, sv.outline)
    if sv.s["shape"].lower() != "cylinder":
        ax.plot(*PROBE, "o", mfc="white", mec="black", ms=8, mew=1.5, zorder=6)
        ax.text(PROBE[0] + 0.06, PROBE[1] + 0.07, f"P: {angle:+.1f}\u00b0", fontsize=11, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(title + "   (time-averaged)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_history(hist, path, t_avg, title):
    t, cd, cl = hist[:, 0], hist[:, 1], hist[:, 2]
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, y, name in ((axs[0], cl, "lift coefficient CL"), (axs[1], cd, "drag coefficient CD")):
        ax.plot(t, y, color="black", lw=1)
        ax.axvspan(t_avg, t[-1], color="0.85", zorder=0)
        ax.set_ylabel(name); ax.grid(alpha=0.3)
        lo, hi = np.percentile(y[t > 1.0], [1, 99]) if (t > 1.0).sum() > 10 else (y.min(), y.max())
        pad = 0.2*(hi - lo) + 1e-3
        ax.set_ylim(lo - pad, hi + pad)
    axs[1].set_xlabel("time (chords, or diameters, travelled)")
    axs[0].set_title(title + "   (shaded: averaging window)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def make_gif(frames, path):
    try:
        from PIL import Image
        imgs = [Image.open(f).convert("RGB").quantize(colors=256) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)
        return True
    except Exception as e:                               # animation is a bonus; never crash for it
        print("  (gif skipped:", e, ")")
        return False


# ------------------------------------------------------------------ RUN ONE CASE
def run(s, label=None):
    tag = label or f"{s['shape']}_a{s['alpha_deg']:g}_Re{s['Re']:g}_{s['mode']}_n{s['cells_per_chord']}"
    out = os.path.join(s["out_dir"], tag)
    os.makedirs(out, exist_ok=True)
    frames_dir = os.path.join(out, "frames")
    if s["frame_every"] > 0:
        os.makedirs(frames_dir, exist_ok=True)

    t_wall = time.time()
    sv = Solver(s)
    print(f"\n{tag}")
    print(f"  grid {sv.nx} x {sv.ny} cells, h = {sv.h:.4f} chord, nu = {sv.nu:.2e}")

    hist, frames, smoke_frames, next_frame, n = [], [], [], 0.0, 0
    t_avg = 0.5*s["t_end"]                               # average over the second half of the run
    sv.t_avg = t_avg
    title = (f"{s['shape']}  alpha = {s['alpha_deg']:g} deg  Re = {s['Re']:,.0f}  "
             f"{s['mode'] if s['shape'].lower() != 'cylinder' else 'validation'}")
    while sv.t < s["t_end"]:
        dt, cd, cl = sv.step()
        n += 1
        hist.append((sv.t, cd, cl))
        if not np.isfinite(cl):
            print("  The run blew up (numbers became infinite). Lower cfl, or use more cells per chord.")
            break
        if s["frame_every"] > 0 and sv.t >= next_frame:
            f = os.path.join(frames_dir, f"vort_{len(frames):04d}.png")
            save_vorticity(sv, f, f"{title}   t = {sv.t:5.2f}")
            frames.append(f)
            f = os.path.join(frames_dir, f"smoke_{len(smoke_frames):04d}.png")
            save_smoke(sv, f, f"{title}   t = {sv.t:5.2f}")
            smoke_frames.append(f); next_frame += s["frame_every"]
        if n % 200 == 0:
            el = time.time() - t_wall
            eta = el / sv.t * (s["t_end"] - sv.t)
            print(f"  t = {sv.t:6.2f} / {s['t_end']:g}   CL = {cl:7.3f}   CD = {cd:7.3f}   "
                  f"dt = {dt:.4f}   about {eta/60:4.1f} min left")

    hist = np.array(hist)
    w = hist[:, 0] >= t_avg
    angle = sv.probe_angle()
    cl_mean, cd_mean = hist[w, 2].mean(), hist[w, 1].mean()
    cl_amp = 0.5*(hist[w, 2].max() - hist[w, 2].min())

    # Strouhal number: the main frequency of the lift wobble (padded FFT, peak refined with a parabola)
    tt = np.linspace(hist[w, 0][0], hist[w, 0][-1], 4096)
    sig = np.interp(tt, hist[w, 0], hist[w, 2]); sig -= sig.mean()
    nfft = 8*len(sig)
    spec = np.abs(np.fft.rfft(sig*np.hanning(len(sig)), nfft))
    k = 1 + np.argmax(spec[1:-1])
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    shift = 0.5*(a - c)/(a - 2*b + c) if (a - 2*b + c) != 0 else 0.0
    st = (k + shift) / (nfft*(tt[1] - tt[0])) if b > 1e-9 else float("nan")

    save_history(hist, os.path.join(out, "forces.png"), t_avg, title)
    save_vorticity(sv, os.path.join(out, "vorticity_final.png"), f"{title}   t = {sv.t:.1f}")
    save_pressure(sv, os.path.join(out, "pressure_streamlines_final.png"), f"{title}   t = {sv.t:.1f}")
    save_smoke(sv, os.path.join(out, "smoke_final.png"), f"{title}   t = {sv.t:.1f}")
    save_mean_streamlines(sv, os.path.join(out, "mean_streamlines.png"), title, angle)
    np.savetxt(os.path.join(out, "history.csv"), hist, delimiter=",", header="t,CD,CL", comments="")
    if s["make_gif"] and len(frames) > 2:
        make_gif(frames, os.path.join(out, "vorticity.gif"))
        make_gif(smoke_frames, os.path.join(out, "smoke.gif"))

    minutes = (time.time() - t_wall)/60
    summary = (f"{title}\n"
               f"grid {sv.nx} x {sv.ny}, {s['cells_per_chord']} cells per chord, t_end = {s['t_end']}\n"
               f"mean CL = {cl_mean:.3f}   (lift wobble +/- {cl_amp:.3f})\n"
               f"mean CD = {cd_mean:.3f}\n"
               f"Strouhal number (main lift frequency x chord / speed) = {st:.3f}\n"
               f"flow angle at P (0.5 chord ahead of, 0.3 chord above the rod) = {angle:+.1f} deg (+ = upwards)\n"
               f"averaged over t = {t_avg:g} to {s['t_end']:g}; run took {minutes:.1f} min\n")
    open(os.path.join(out, "summary.txt"), "w").write(summary)
    print("  " + summary.replace("\n", "\n  "))
    print(f"  pictures and numbers saved in {out}/")
    return dict(alpha=s["alpha_deg"], CL=cl_mean, CD=cd_mean, CL_wobble=cl_amp, St=st,
                flow_angle_P=angle, folder=out)


# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser(description="2D Navier-Stokes around an aerofoil")
    ap.add_argument("--quick", action="store_true", help="coarse, short run to check everything works")
    ap.add_argument("--validate", action="store_true", help="cylinder at Re 100 vs published values")
    ap.add_argument("--sweep", action="store_true", help="run every angle in SWEEP_ALPHAS")
    ap.add_argument("--shape"); ap.add_argument("--alpha", type=float); ap.add_argument("--re", type=float)
    ap.add_argument("--mode", choices=["tunnel", "free"]); ap.add_argument("--res", type=int)
    ap.add_argument("--tend", type=float); ap.add_argument("--colour", action="store_true")
    ap.add_argument("--height", type=float, help="tunnel mode: chamber height in chords (default 2.0)")
    a = ap.parse_args()

    s = dict(SETTINGS)
    if a.quick:
        s.update(cells_per_chord=48, t_end=4.0, frame_every=0.5)
    if a.validate:
        s.update(shape="cylinder", alpha_deg=0.0, Re=100, cells_per_chord=24, t_end=80.0,
                 frame_every=1.0, mode="free")
    for key, val in (("shape", a.shape), ("alpha_deg", a.alpha), ("Re", a.re), ("mode", a.mode),
                     ("cells_per_chord", a.res), ("t_end", a.tend), ("tunnel_height", a.height)):
        if val is not None:
            s[key] = val
    if a.colour:
        s["colour"] = True

    if a.sweep:
        rows = []
        s["frame_every"] = 0.0                           # no animation frames in a sweep: faster
        for al in SWEEP_ALPHAS:
            s["alpha_deg"] = float(al)
            rows.append(run(s))
            path = os.path.join(s["out_dir"], f"sweep_{s['shape']}_Re{s['Re']:g}_{s['mode']}.csv")
            with open(path, "w", newline="") as f:      # rewritten after every angle, so nothing is lost
                wr = csv.DictWriter(f, fieldnames=["alpha", "CL", "CD", "CL_wobble", "St", "flow_angle_P", "folder"])
                wr.writeheader(); wr.writerows(rows)
        al = np.array([r["alpha"] for r in rows]); cl = np.array([r["CL"] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(al, cl, "ko-", label=f"simulation ({s['mode']} mode)")
        ax.plot(al, 2*np.pi*np.radians(al), "k--", lw=1, label="thin-aerofoil theory, 2 pi alpha")
        ax.set_xlabel("angle of attack (deg)"); ax.set_ylabel("lift coefficient CL")
        ax.set_title(f"{s['shape']}  Re = {s['Re']:,.0f}"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(s["out_dir"], f"lift_curve_{s['shape']}_Re{s['Re']:g}_{s['mode']}.png"), dpi=120)
        plt.close(fig)
        ang = np.array([r["flow_angle_P"] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(al, ang, "ko-", label="simulation")
        ax.set_xlabel("angle of attack (deg)"); ax.set_ylabel("flow angle at P (deg, positive = upwards)")
        ax.set_title(f"{s['shape']}  Re = {s['Re']:,.0f}: add your smoke readings as dots")
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(s["out_dir"], f"flow_angle_{s['shape']}_Re{s['Re']:g}_{s['mode']}.png"), dpi=120)
        plt.close(fig)
        print(f"\nSweep done: {path}")
        return

    r = run(s)
    if a.validate:
        print("\nPublished values for a cylinder at Re = 100 in open flow:")
        print("  Strouhal number about 0.164, mean drag coefficient about 1.33 to 1.35.")
        print("  The walls (5% blockage) and the coarse grid push both up a little; within about 10% is a pass.")
        print(f"  Yours: St = {r['St']:.3f}, CD = {r['CD']:.3f}")


if __name__ == "__main__":
    main()
