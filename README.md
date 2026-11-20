# navier-stokes-2d

A Python program that simulates air flowing past a wing shape (an aerofoil) or a cylinder. It works out the lift, the drag and the swirls (vortices) that form behind the shape, by solving the Navier–Stokes equations step by step on a grid.

![Vortex street behind a cylinder at Re = 100](docs/vortex_street_cylinder_Re100.gif)

*Swirls peeling off behind a cylinder, first from one side, then the other. This pattern is called a von Kármán vortex street.*

## The equations

The Navier–Stokes equations are Newton's second law, $F = ma$, written for a tiny blob of fluid:

$$\underbrace{\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}}_{\text{acceleration}} = \underbrace{-\nabla p}_{\text{pressure push}} + \underbrace{\frac{1}{Re}\nabla^2\mathbf{u}}_{\text{friction}}$$

$$\nabla\cdot\mathbf{u} = 0 \quad\text{(air doesn't pile up anywhere)}$$

What each symbol means, using multivariable calculus:

- $\mathbf{u} = (u, v)$ is the velocity at each point: $u$ is the speed to the right, $v$ is the speed upwards. It is a vector field, so every point $(x, y)$ has its own arrow.
- $p$ is the pressure, a scalar field.
- $\partial \mathbf{u}/\partial t$ is how fast the velocity changes at one fixed point.
- $(\mathbf{u}\cdot\nabla)\mathbf{u} = u\,\dfrac{\partial \mathbf{u}}{\partial x} + v\,\dfrac{\partial \mathbf{u}}{\partial y}$ is the extra acceleration a blob gets by moving into a region where the air is faster or slower.
- $-\nabla p = -\left(\dfrac{\partial p}{\partial x}, \dfrac{\partial p}{\partial y}\right)$: the gradient points uphill in pressure, so the minus sign means air is pushed from high pressure to low.
- $\nabla^2\mathbf{u} = \dfrac{\partial^2 \mathbf{u}}{\partial x^2} + \dfrac{\partial^2 \mathbf{u}}{\partial y^2}$ (the Laplacian) compares a point with the average of its neighbours. Friction (viscosity) evens out those differences.
- $\nabla\cdot\mathbf{u} = \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y}$ (the divergence) is the net flow out of a tiny box. Setting it to 0 means whatever flows in must flow out.
- $Re$, the **Reynolds number**, compares how hard the air pushes on with how sticky it is. Small $Re$ is like honey; large $Re$ is like air round a jet. Everything here uses $Re = 1{,}500$, which is roughly a 3 cm wing in a gentle breeze.

Units are scaled so the wing's length (the chord), the air speed and the air density are all 1. Time is measured in "how many chord lengths the air has travelled".

## How the program solves them

Nobody can solve these equations with a formula for a real shape, so the computer does it numerically:

1. **Chop space into a grid of small squares.** Pressure is stored in the middle of each square and velocity on its edges (a "staggered grid"). That way the computer can check exactly how much air enters and leaves every square.

   ![Staggered grid](docs/staggered_grid.png)

2. **Turn derivatives into differences.** A derivative like $\partial u/\partial x$ becomes (value in the next square − value in the previous square) ÷ (distance between them). This is the limit definition of a derivative, without taking the limit.

3. **Step forward in time.** Each time step:
   - **Guess:** move the air forward using only its own motion and friction. Call this guess $\mathbf{u}^*$.
   - **Stop the air inside the wing:** set the velocity to zero inside the shape.
   - **Fix the guess with pressure:** the guess makes air pile up in some squares. Pressure is whatever is needed to undo that. The computer solves
     $$\nabla^2\phi = \nabla\cdot\mathbf{u}^*$$
     and then subtracts $\nabla\phi$ from the velocity. Now $\nabla\cdot\mathbf{u} = 0$ everywhere.
   
   It does this three times per step and averages them in a clever way (a method called RK3), which makes it more accurate.

4. **Keep the steps small.** Each step is short enough that the air never jumps more than half a square at a time. Otherwise the simulation blows up.

**The pressure trick.** Solving for pressure means solving about 49,000 equations at once. Instead of doing it by brute force, the program writes the answer as a sum of cosine waves (a cosine transform, like a Fourier series). For cosine waves the equation becomes a simple division, so the exact answer comes out in a fraction of a second.

**The wing on the grid.** The wing isn't drawn with smooth curves. Each square is marked with how much of it is inside the wing, from 0 (all air) to 1 (all wing), so the outline looks like a staircase.

![The wing drawn on the grid](docs/mask.png)

**Lift and drag.** Every step, the program stops the air inside the wing. The push needed to stop it is equal and opposite to the force the air puts on the wing (Newton's third law). Adding those pushes up gives lift (upwards) and drag (backwards). They are reported as coefficients:

$$C_L = \frac{\text{lift}}{\tfrac12\rho U^2 c}, \qquad C_D = \frac{\text{drag}}{\tfrac12\rho U^2 c}$$

so the numbers don't depend on the wing's size or the air speed.

**Other outputs.**
- **Shedding frequency:** how often a swirl leaves the wing, written as the Strouhal number $St = fc/U$.
- **Dye ("smoke"):** imaginary dye released upstream and carried by the air, to show where it goes.
- **Point P:** a spot just in front of and above the wing, where the program measures the average angle at which the air rises.

## Is it right?

| Test | This code | Real answer |
|---|---|---|
| Cylinder at Re 100: how often swirls leave | 0.176 | 0.164–0.165 |
| Cylinder at Re 100: drag coefficient | 1.46 | 1.33–1.36 |
| Symmetric wing at 0°: lift | 0.000 | exactly 0 |

The cylinder answers are 7–9% too high. The walls of the simulated box and the fairly large grid squares both push them up.

**Does it change if the grid gets finer?** At $Re = 3{,}000$, going from 64 to 100 squares per chord changed the lift by only 4%, which is good. At $Re = 20{,}000$ it keeps changing (lift 0.28, 0.66, 0.83 at 64, 100, 150 squares), so the program can't be trusted there. The thin layer of slow air on the wing's surface becomes thinner than a few squares.

## Results: two wings at Re 1,500

The NACA 2412 is slightly curved (cambered); the NACA 0012 is symmetric. Box walls 2 chords apart, 64 squares per chord. The raw numbers are in [`examples/`](examples).

| Angle | 2412 lift $C_L$ | 2412 drag $C_D$ | 0012 lift $C_L$ | 0012 drag $C_D$ |
|---|---|---|---|---|
| 0° | 0.003 | 0.110 | 0.000 | 0.107 |
| 4° | 0.228 | 0.126 | 0.183 | 0.119 |
| 8° | 0.444 | 0.185 | 0.338 | 0.179 |
| 12° | 0.707 | 0.295 | 0.586 | 0.286 |
| 16° | 1.002 | 0.466 | 0.858 | 0.446 |
| 20° | 1.272 | 0.652 | 1.151 | 0.653 |

![Lift curve](docs/lift_curve_NACA2412_Re1500.png)

Lift grows with angle, but only about half as fast as textbook theory predicts. That theory ignores friction; at this small, slow scale friction matters a lot, and the air stops hugging the wing early. The curved 2412 makes more lift than the symmetric 0012 at every angle above 0°.

![Vorticity, NACA 2412 at 16 degrees](docs/vorticity_NACA2412_16deg.png)

*The 2412 at 16°: the air peels off the top and sheds big swirls.*

![Pressure and streamlines, NACA 2412 at 8 degrees](docs/pressure_NACA2412_8deg.png)

*Pressure at 8°: dark means low pressure on top of the wing. That suction is most of the lift.*

![Dye lines, NACA 2412 at 8 degrees](docs/dye_NACA2412_8deg.png)

*Dye at 8°, rolling up into swirls behind the wing.*

## Running it

Needs Python 3.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Command | What it does | Time* |
|---|---|---|
| `python3 ns_aerofoil.py --quick` | quick low-detail test | 15 s |
| `python3 ns_aerofoil.py --alpha 8 --res 64` | one wing at 8° | 1–2 min |
| `python3 ns_aerofoil.py` | default settings, 100 squares per chord | about 9 min |
| `python3 ns_aerofoil.py --sweep --res 64` | angles 0° to 20°, with a lift graph | about 8 min |
| `python3 ns_aerofoil.py --validate` | the cylinder test | about 13 min |

*On a cloud computer. A laptop can take up to twice as long.

Settings you can change (they can be combined):

| Option | Meaning | Default |
|---|---|---|
| `--shape` | `NACA2412`, `NACA0012`, any 4-digit NACA wing, or `cylinder` | `NACA2412` |
| `--alpha` | angle of the wing, in degrees | 6 |
| `--re` | Reynolds number | 1500 |
| `--mode` | `tunnel` (walls close by) or `free` (walls 6 chords apart) | `tunnel` |
| `--height` | distance between the walls in tunnel mode, in chords | 2.0 |
| `--res` | squares per chord (more = sharper but slower) | 100 |
| `--tend` | how long to simulate, in chord lengths travelled | 16 |
| `--colour` | colour pictures instead of grey | off |

Example: `python3 ns_aerofoil.py --shape cylinder --re 200 --mode free --res 32 --tend 60`

## What it saves

Each run goes into its own folder, `results/<shape>_a<angle>_Re<Re>_<mode>_n<res>/`:

| File | What it shows |
|---|---|
| `vorticity.gif`, `vorticity_final.png` | how fast the air is spinning at each point |
| `smoke.gif`, `smoke_final.png` | the dye lines |
| `pressure_streamlines_final.png` | pressure, with lines showing the air's path |
| `mean_streamlines.png` | the average flow over time |
| `forces.png`, `history.csv` | lift and drag over time |
| `summary.txt` | the final numbers |

To play a GIF on a Mac, click it in Finder and press the space bar.

## Limits

- It's 2D, like looking at a slice of an endless wing. Real wings have tips, and swirls form there too.
- The wing outline is a staircase of squares, not a smooth curve.
- It doesn't model turbulence (chaotic, messy flow), so it only works for slow, small, smooth flows.
- Right at the wing's surface it is less accurate than everywhere else.

## Files

```
ns_aerofoil.py      the program
requirements.txt    the Python libraries it needs
docs/               pictures used in this README
examples/           the results for angles 0° to 20°
```
