# navier-stokes-2d
# 2D Navier–Stokes solver

A finite-volume solver for the 2D incompressible Navier–Stokes equations, written from scratch in Python (NumPy and SciPy). It simulates flow past NACA aerofoils and a circular cylinder, and computes lift, drag and vortex shedding.

![Vortex street behind a cylinder at Re = 100](docs/vortex_street_cylinder_Re100.gif)

*Vorticity behind a cylinder at Re = 100: the von Kármán vortex street.*

 The equations

Non-dimensional incompressible Navier–Stokes, with chord length, free-stream speed and density all equal to 1:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2\mathbf{u} + \mathbf{f}$$

$$\nabla\cdot\mathbf{u} = 0$$

$\mathbf{u} = (u, v)$ is velocity, $p$ is pressure, $Re = Uc/\nu$ is the Reynolds number and $\mathbf{f}$ is the forcing that holds the fluid still inside the body. Time is measured in chords travelled by the free stream.

 Method

**Staggered (MAC) grid.** Pressure sits at cell centres, $u$ on vertical faces and $v$ on horizontal faces. Each velocity sits exactly on the face it carries flow through, so the mass balance of every cell is exact and pressure can't decouple into a checkerboard.

![Staggered grid](docs/staggered_grid.png)

**Projection method.** Each step first moves the velocity forward using advection and viscosity only, giving $\mathbf{u}^*$. This velocity is not divergence-free, so a pressure correction is solved for and subtracted:

$$\nabla^2\phi = \frac{\nabla\cdot\mathbf{u}^*}{\Delta t}, \qquad \mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t\,\nabla\phi$$

**Boundaries.** Uniform inflow on the left, walls with no flow through them at the top and bottom, and an outflow on the right scaled so it carries exactly the mass that comes in.

**Pressure Poisson solve.** All boundaries have Neumann conditions, so the Laplacian diagonalises under a 2D discrete cosine transform. The Poisson equation is solved exactly in $O(N \log N)$ with `scipy.fft.dctn`, with no iteration.

**Advection and diffusion.** Third-order upwind differences for $(\mathbf{u}\cdot\nabla)\mathbf{u}$, second-order central differences for $\nabla^2\mathbf{u}$.

**Time stepping.** Three-stage strong-stability-preserving Runge–Kutta (SSP-RK3), with a projection at every stage. The time step is set each step by a CFL limit (default 0.5) and the viscous stability limit.

**Immersed boundary.** The body is not meshed. Each cell gets a solid fraction $\chi$ between 0 (fluid) and 1 (solid), and the velocity is forced towards zero in proportion to $\chi$ (direct forcing).

![Solid fraction mask on the grid](docs/mask.png)

**Forces.** The momentum the mask removes from the fluid each step equals the force the body exerts on the flow, so summing it gives lift and drag, weighted by the RK3 stage weights (1/6, 2/3, 1):

$$C_L = \frac{F_y}{\tfrac12\rho U^2 c}, \qquad C_D = \frac{F_x}{\tfrac12\rho U^2 c}$$

**Shedding frequency.** The Strouhal number $St = fc/U$ comes from the dominant frequency in the lift history.

**Passive dye.** Lines of dye are released upstream and carried by the flow (streaklines). The code calls these "smoke".

**Probe point P.** A point 0.5 chord ahead of and 0.3 chord above the pivot records the time-averaged flow angle there (upwash), which rises with lift.

 Validation

| Test | This code | Published |
|---|---|---|
| Cylinder, Re 100: Strouhal number | 0.176 | 0.164–0.165 |
| Cylinder, Re 100: mean drag coefficient | 1.46 | 1.33–1.36 |
| NACA 0012 at 0°: lift coefficient | 0.000 | 0 (symmetric section) |

The cylinder results are 7–9% high. The side walls (5% blockage) and the coarse grid (24 cells per diameter) both push them up.

**Grid convergence.** At Re 3,000, refining from 64 to 100 cells per chord changed lift by 4% and the probe angle by 0.1°. At Re 20,000 the lift has not converged (CL = 0.28, 0.66 and 0.83 at 64, 100 and 150 cells per chord), so results above a few thousand should not be trusted at these grid sizes.

 Results: NACA 2412 and NACA 0012 at Re 1,500

Channel walls 2 chords apart, 64 cells per chord. The raw data is in [`examples/`](examples).

| α | 2412 C_L | 2412 C_D | 0012 C_L | 0012 C_D |
|---|---|---|---|---|
| 0° | 0.003 | 0.110 | 0.000 | 0.107 |
| 4° | 0.228 | 0.126 | 0.183 | 0.119 |
| 8° | 0.444 | 0.185 | 0.338 | 0.179 |
| 12° | 0.707 | 0.295 | 0.586 | 0.286 |
| 16° | 1.002 | 0.466 | 0.858 | 0.446 |
| 20° | 1.272 | 0.652 | 1.151 | 0.653 |

![Lift curve](docs/lift_curve_NACA2412_Re1500.png)

The lift slope is about half the thin-aerofoil value of 2π per radian. That is expected at Re 1,500: the boundary layers are thick, viscous drag is large and the flow separates early. The cambered 2412 gives more lift than the symmetric 0012 at every angle.

![Vorticity, NACA 2412 at 16 degrees](docs/vorticity_NACA2412_16deg.png)

*NACA 2412 at 16°: the flow separates and sheds vortices.*

![Pressure and streamlines, NACA 2412 at 8 degrees](docs/pressure_NACA2412_8deg.png)

*Pressure coefficient and streamlines at 8°: low pressure over the upper surface.*

![Dye lines, NACA 2412 at 8 degrees](docs/dye_NACA2412_8deg.png)

*Dye lines at 8°, rolling up into the wake.*

 Running it

Needs Python 3.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Command | What it does | Time* |
|---|---|---|
| `python3 ns_aerofoil.py --quick` | coarse grid, short run: checks everything works | 15 s |
| `python3 ns_aerofoil.py --alpha 8 --res 64` | one aerofoil case | 1–2 min |
| `python3 ns_aerofoil.py` | default settings (100 cells per chord) | about 9 min |
| `python3 ns_aerofoil.py --sweep --res 64` | 0° to 20°: CSV, lift curve | about 8 min |
| `python3 ns_aerofoil.py --validate` | cylinder at Re 100 against published values | about 13 min |

*On a cloud Linux machine. A laptop may take up to twice as long.

Options can be combined:

| Option | Meaning | Default |
|---|---|---|
| `--shape` | `NACA2412`, `NACA0012`, any NACA 4-digit, or `cylinder` | `NACA2412` |
| `--alpha` | angle of attack in degrees | 6 |
| `--re` | Reynolds number | 1500 |
| `--mode` | `tunnel` (walls `--height` chords apart) or `free` (walls 6 chords apart) | `tunnel` |
| `--height` | channel height in chords, tunnel mode | 2.0 |
| `--res` | cells per chord | 100 |
| `--tend` | simulated time in chords travelled | 16 |
| `--colour` | colour plots instead of greyscale | off |

Example: `python3 ns_aerofoil.py --shape cylinder --re 200 --mode free --res 32 --tend 60`

 Output

Each run saves to `results/<shape>_a<angle>_Re<Re>_<mode>_n<res>/`:

| File | Contents |
|---|---|
| `vorticity.gif`, `vorticity_final.png` | vorticity field |
| `smoke.gif`, `smoke_final.png` | passive dye lines |
| `pressure_streamlines_final.png` | pressure coefficient with streamlines |
| `mean_streamlines.png` | time-averaged streamlines |
| `forces.png`, `history.csv` | lift and drag against time |
| `summary.txt` | mean C_L and C_D, Strouhal number, probe angle |

To watch a GIF on a Mac, select it in Finder and press the space bar.

 Limits

- 2D only, so there are no wing-tip vortices and no 3D breakdown of the wake.
- The body is a staircase on a square grid, so the surface is only as smooth as the grid.
- No turbulence model: results are only reliable at low Reynolds numbers (up to a few thousand at these grid sizes).
- First-order accurate at the immersed boundary, even though the scheme is higher order away from it.

 Files

```
ns_aerofoil.py      the solver
requirements.txt    numpy, scipy, matplotlib, pillow
docs/               figures used in this README
examples/           angle sweep data at Re 1,500
```
