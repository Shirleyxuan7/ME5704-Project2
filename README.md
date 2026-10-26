# ME5704 Project 2 — Heat conduction in a plate with an internal line source

Numerical solution of all three parts of the assignment:

| Part | Question | Main method | Cross-checks |
|---|---|---|---|
| (a) | Steady-state *T* at the 9 interior points (Laplace's equation) | 5-point finite differences + unequal-spacing stencil at the off-grid source, solved by Gauss elimination / SOR with optimal λ | Jacobi, Gauss-Seidel (Liebmann), LU; grid refinement h = 1 → 1/64 with Richardson extrapolation; 3 ways of modelling the source |
| (b) | Heat-flux density **q** = −k∇T | Source-aware central differences O(h²) | forward/backward O(h) and O(h²), naive central; fine-grid + Richardson reference; energy balance through 3 control surfaces |
| (c) | Heat absorbed h = ∫ qA dt for 0 ≤ t ≤ 4 | h = Φ · ∫k dt, ∫k dt by Simpson 1/3 | trapezoid, Simpson 3/8, Romberg, polynomial regression, natural cubic spline, Lagrange + Gauss-Legendre, Gauss-Newton fit of k = a − b·e^(−ct) |

Everything (every number, table and figure used in the report) is produced by one command.

> **At a glance:** `python main.py` generates every result in `results/`; `pytest` runs all 39 tests.
> The header of each module in `heatplate/` states the Lecture Notes section it implements.
> Final answers are in `results/summary.md`.

**Team (NUS ME5704, Group __):** _Name 1_ · _Name 2_ · _Name 3_ · _Name 4_ — developed collaboratively;
see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

---

## 1. Quick start

```bash
# Python >= 3.9
pip install -r requirements.txt

python main.py            # full run, grids down to h = 1/64  (~15 s)
python main.py --fast     # grids down to h = 1/32            (~3 s)
python main.py --no-plots # tables + summary only

pytest -v                 # 39 tests, < 2 s
```

Outputs:

```
results/
├── summary.md            <- final answers for (a), (b), (c) in report-ready tables
├── tables/               <- every intermediate result as CSV
│   ├── a1_solvers_h1.csv            all solvers on the 9x9 system, iterations, agreement
│   ├── a2_lambda_sweep.csv          spectral radius and iteration count vs lambda
│   ├── a3_grid_refinement.csv       9-point T on h = 1 ... 1/64, observed order, Richardson
│   ├── a4_source_modelling.csv      exact (unequal stencil) vs source snapped to x = 1 / x = 2
│   ├── a5_iterations_vs_grid.csv    cost of Jacobi / Gauss-Seidel / SOR vs grid size
│   ├── a6_verification_sinh.csv     error vs h for a smooth analytic test case
│   ├── b1_flux_h1_all_schemes.csv   q/k at the 9 points for 6 difference schemes + error
│   ├── b2_flux_reference.csv        grid-converged q/k (reference)
│   ├── b3_heat_flow_phi.csv         heat flow Phi through 3 surfaces vs h
│   ├── c1_integral_k.csv            int_0^4 k dt by 15 methods
│   └── c2_heat_absorbed.csv         final h
└── figures/
    ├── fig1_temperature_field.png   contour map with h = 1 and converged values
    ├── fig2_solver_convergence.png  error history; rho(lambda) + golden/parabolic search; iterations vs lambda
    ├── fig3_iterations_vs_grid.png  O(n^2) vs O(n) growth of iteration counts
    ├── fig4_grid_convergence.png    log-log error vs h: p ~ 1 (assignment) vs p ~ 2 (smooth test)
    ├── fig5_source_modelling.png    error of the three source treatments at h = 1
    ├── fig6_heat_flux.png           flux vectors on the temperature field
    ├── fig7_heat_flow_phi.png       energy balance: Phi through edges / box / source faces
    ├── fig8_k_fits.png              k(t) data and fitted models
    └── fig9_k_integrals.png         int k dt by method
```

---

## 2. Repository layout

| File | Content | Lecture Notes |
|---|---|---|
| `heatplate/config.py` | All problem data (geometry, BCs, k-data, tolerances) | – |
| `heatplate/discretisation.py` | Builds the FD equations; unequal-spacing stencil; 3 source modes; matrix assembly | 6.3.1 |
| `heatplate/linear_solvers.py` | Gauss elimination (partial pivoting), LU, Jacobi, Gauss-Seidel, SOR | 3.5.2, 6.3.1 |
| `heatplate/relaxation.py` | Spectral radius of SOR, golden-section & parabolic-interpolation search for λ_opt, Young's formula | 4.2, 6.3.1 |
| `heatplate/grid_solver.py` | Matrix-free red-black Jacobi / Gauss-Seidel / SOR (Liebmann) for fine grids | 6.3.1 |
| `heatplate/grid_study.py` | Grid refinement, observed order, generalised Richardson, analytic test fields | 5.3.2, 6.3.1 |
| `heatplate/finite_difference.py` | General FD weights (any spacing), source-aware derivatives | 5.5.1–5.5.3 |
| `heatplate/flux.py` | q/k at the 9 points, heat flow Φ through outer edges / box / source faces | 5.2, 5.5 |
| `heatplate/integration.py` | Trapezoid, Simpson 1/3 & 3/8, mixed, Romberg (data & function), Gauss-Legendre 1–6 pts | 5.2, 5.3 |
| `heatplate/curve_fit.py` | Polynomial regression, Lagrange, natural cubic spline, Gauss-Newton | 2.2, 2.3 |
| `heatplate/heat_absorbed.py` | All ∫k dt estimates; h = Φ·∫k dt | 5, 2 |
| `heatplate/plotting.py` | All figures | – |
| `main.py` | Runs parts (a)→(b)→(c), writes tables, figures and `summary.md` | – |
| `tests/` | One test file per module, each with a known-answer case | – |

**All numerical algorithms are written by hand.** numpy is used for array storage and
vectorised arithmetic; `numpy.linalg.eigvals` is only used to *analyse* the SOR iteration
matrix; scipy is only used inside the tests as an independent reference.

Conventions: `T[i, j]` with x = i·h, y = j·h; 3×3 outputs are laid out like the sketch (top row y = 3, left column x = 1).

---

## 3. Modelling decisions (agreed assumptions)

1. **Off-grid source.** On the h = 1 grid the red line x = 1.5 lies halfway between the columns
   x = 1 and x = 2. For the four nodes (1,2), (2,2), (1,3), (2,3) the horizontal neighbour across
   the line is replaced by the source point (distance 0.5 h, T = 350 K) using the unequal-spacing formula

   ∂²T/∂x² ≈ 2/h² · [ T_W/(α_W(α_W+α_E)) + T_E/(α_E(α_W+α_E)) − T_P/(α_W α_E) ],

   e.g. for (1,2): (4/3)·200 + (8/3)·350 + T(1,1) + T(1,3) − 6·T(1,2) = 0.
   On grids with h = 1/2, 1/4, … the line falls on grid nodes, which are simply fixed at 350 K.
   The end points (1.5, 2) and (1.5, 3) are treated as part of the source.
2. **Quasi-steady state.** k does not appear in Laplace's equation, so T(x, y) is independent of k and
   q(x, y, t) = k(t)·(−∇T). Valid when k changes slowly compared with the diffusion time of the plate.
3. **Meaning of qA.** The heat flow through a closed surface around the source,
   Q(t) = ∮ q·n dA = k(t)·Φ, with dA = ds × thickness (thickness = 1). Hence h = Φ·∫₀⁴ k dt.
   At steady state the plate stores no net energy: the heat taken from the source equals the heat leaving
   through the edges — this is used as a check (Φ through different surfaces must agree).
4. **Units.** The assignment does not give units for k and t, so h is reported in "k-units × K × length".

---

## 4. Results (from `results/summary.md`)

**(a) Temperature [K]** — the h = 1 answer requested by the assignment, and the grid-converged estimate:

| y \ x | 1 | 2 | 3 |   | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| 3 | 280.87 | 289.61 | 231.48 | | 266.19 | 272.40 | 224.13 |
| 2 | 285.19 | 295.67 | 236.31 | | 271.50 | 279.39 | 229.01 |
| 1 | 230.30 | 236.01 | 218.08 | | 226.48 | 231.35 | 215.99 |
|   | *h = 1* | | | | *converged (h → 0)* | | |

**(b) Heat flux** (h = 1): e.g. q(1,2) = k·(−114.8, −25.3), q(2,2) = k·(92.2, −26.8); full table in `summary.md`.

**(c) Heat absorbed:** Φ = 460.8 (converged; 585.9 on the h = 1 grid), ∫₀⁴ k dt = 1.2733 (Simpson 1/3)
→ **h ≈ 587** (≈ 746 if the coarse h = 1 grid is used for Φ).

---

## 5. Findings worth discussing in the report

* **All solvers agree** to round-off on the 9×9 system; iterations to ε_s = 10⁻⁸ %: Jacobi 42, Gauss-Seidel 23,
  SOR 14. For 9 unknowns a direct solver is the sensible choice — iterative methods only pay off on fine grids,
  where Jacobi/GS cost grows like n² and SOR like n (fig. 3).
* **Optimal λ by 1-D optimisation.** Golden-section search reproduces Young's λ_opt = 1.0987 in 29 evaluations;
  parabolic interpolation needs ~100 because ρ(λ) has a kink at the optimum (it assumes a smooth function).
* **The coarse grid is far from converged:** h = 1 overestimates T by up to 17 K and Φ by 27 %.
* **Observed order ≈ 1, not 2.** The same code converges at p = 1.99 on a smooth analytic test, but only p ≈ 1 on
  the assignment because the ends of the line source are singular (T ~ r^½). Richardson extrapolation must therefore
  use the observed p rather than the textbook p = 2.
* **Source modelling matters more than the solver:** snapping the source to x = 1 or x = 2 gives errors up to 80 K (fig. 5).
* **Flux next to the source:** a central difference that reaches across the line (`naive_central`) is physically
  meaningless because ∇T is discontinuous there; source-aware stencils are needed. Local q at the line tips does not
  converge (it is infinite), but the integrated Φ does.
* **Energy balance:** Φ through the plate edges and through a box around the source agree to 0.01 % on the finest grid;
  integrating directly on the source faces converges very slowly (p ≈ 0.4) because of the tip singularity.
* **Time integration is not the bottleneck:** all ∫k dt methods that use all five data points agree within 1.2 %
  (1.260–1.275), whereas the grid changes Φ by 27 %. With only 5 noisy-looking data points, a higher-order rule is not
  automatically "more accurate" — the spread between methods reflects data uncertainty. Note that the cubic and
  quadratic regressions give the *same* integral: for points symmetric about t = 2 the extra cubic term is odd and
  integrates to zero.

---

## 6. Tests (known-answer cases)

| Test file | Known answer used |
|---|---|
| `test_linear_solvers.py` | Lecture 6.3.1 exercise (λ = 1.1, ε_s = 1 % → 46.9, 40.6, 71.9, 65.6 after 4 iterations); random diagonally-dominant systems vs `numpy.linalg.solve`; pivoting case; coefficients of the irregular stencil |
| `test_relaxation.py` | Minimum of (x−2)²; Young's formula 2/(1+sin(π/4)) for the plate without source |
| `test_grid.py` | T = 300 + (x²−y²)/2 is reproduced *exactly* (on- and off-grid source); smooth sinh field converges at p ≈ 2; grid solver = direct solver = scipy; maximum principle |
| `test_flux.py` | Textbook FD weights; exact gradients for a quadratic field; zero net flux for a source-free field; energy balance of the assignment |
| `test_integration.py` | Lecture example ∫₁² x⁷ dx = 31.875 (Gauss 2/3/4-pt = 31.201/31.871/31.875, trapezoid 40.793/34.155, Romberg); degree of precision of each rule; recovery of known regression / Gauss-Newton parameters |

---

## 7. Extending / changing parameters

* Change the problem data in `heatplate/config.py` (e.g. a different source temperature or k-data).
* `GRID_SPACINGS` controls the refinement study; any h with 4/h integer works (odd 1/h puts the source off-grid again).
* Source treatment: `discretise(problem, h, source_mode="exact" | "snap_left" | "snap_right")`.
* New test field: build a `Problem(boundary_T=f, source_T=f)` with any harmonic `f(x, y)`.
