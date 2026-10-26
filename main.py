"""ME5704 Project 2 - run everything and reproduce every table / figure of the report.

    python main.py            # full run (grids down to h = 1/64, ~10-20 s)
    python main.py --fast     # grids down to h = 1/32 only
    python main.py --no-plots

Outputs:  results/tables/*.csv,  results/figures/*.png,  results/summary.md
"""
import argparse
import csv
import os
import time

import numpy as np

from heatplate import config, plotting
from heatplate.discretisation import assignment_problem, discretise, sample
from heatplate.flux import flux_at_points, flux_field, phi_box, phi_outer, phi_source
from heatplate.grid_solver import solve_grid
from heatplate.grid_study import (extrapolate_last_three, harmonic_test_problem, observed_order,
                                  refinement_study, richardson)
from heatplate.heat_absorbed import heat_absorbed, integrate_k_all
from heatplate.linear_solvers import (gauss_elimination, gauss_seidel, is_diagonally_dominant, jacobi,
                                      lu_decompose, lu_solve, sor)
from heatplate.relaxation import optimise_lambda, sor_spectral_radius, sweep_iterations

ROOT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
PTS = [f"({x},{y})" for y in (3, 2, 1) for x in (1, 2, 3)]       # sketch layout, top row first


def write_csv(name, header, rows):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in r])


def grid3(a):
    return "\n".join("    " + "  ".join(f"{v:8.2f}" for v in row) for row in a)


def md_grid(a, fmt="{:.2f}"):
    s = "| y \\ x | 1 | 2 | 3 |\n|---|---|---|---|\n"
    for r, yy in enumerate((3, 2, 1)):
        s += f"| {yy} | " + " | ".join(fmt.format(v) for v in a[r]) + " |\n"
    return s


def hlabel(h):
    return "1" if h == 1 else f"1/{round(1 / h)}"


# =============================================================================== PART A
def part_a(hs, plots):
    print("\n=== PART (a): steady-state temperature ===")
    prob = assignment_problem()
    disc = discretise(prob, 1.0, "exact")
    A, b, nodes = disc.to_matrix()
    print(f"  9x9 system assembled (unequal-spacing stencil next to the source); "
          f"diagonally dominant: {is_diagonally_dominant(A)}")

    # --- 1. all linear solvers on the h = 1 system
    eps = 1e-8
    lam_info = optimise_lambda(A)
    lam_opt = lam_info["lam_golden"]
    x_gauss = gauss_elimination(A, b)
    Lm, U = lu_decompose(A)
    x_lu = lu_solve(Lm, U, b)
    r_j = jacobi(A, b, eps_s=eps)
    r_gs = gauss_seidel(A, b, eps_s=eps)
    r_sor = sor(A, b, lam_opt, eps_s=eps)
    r_sor11 = sor(A, b, 1.1, eps_s=1.0)                     # lecture-style settings for comparison
    solver_rows = [("Gauss elimination", "-", x_gauss), ("LU decomposition", "-", x_lu),
                   ("Jacobi", r_j.iterations, r_j.x), ("Gauss-Seidel (Liebmann)", r_gs.iterations, r_gs.x),
                   (f"SOR lam={lam_opt:.4f}", r_sor.iterations, r_sor.x),
                   ("SOR lam=1.1, eps_s=1% (lecture)", r_sor11.iterations, r_sor11.x)]
    write_csv("a1_solvers_h1.csv", ["method", "iterations"] + [f"T{n}" for n in nodes] + ["max|T-T_gauss|"],
              [[m, it] + list(x) + [float(np.max(np.abs(x - x_gauss)))] for m, it, x in solver_rows])
    for m, it, x in solver_rows:
        print(f"    {m:34s} iterations={str(it):>4s}  max|T - T_Gauss| = {np.max(np.abs(x - x_gauss)):.2e}")
    print(f"  lambda: Young={lam_info['lam_young']:.5f}, golden={lam_info['lam_golden']:.5f} "
          f"({lam_info['evals_golden']} evals), parabolic={lam_info['lam_parabolic']:.5f} "
          f"({lam_info['evals_parabolic']} evals)")
    T_h1 = disc.to_field(x_gauss)
    T9_h1 = sample(T_h1, disc)

    lams = np.linspace(1.0, 1.95, 191)
    rhos = np.array([sor_spectral_radius(A, l) for l in lams])
    iters = sweep_iterations(A, b, lams, eps_s=eps)
    write_csv("a2_lambda_sweep.csv", ["lambda", "spectral_radius", "iterations"], zip(lams, rhos, iters))

    # --- 2. grid refinement with the heat-flow functionals needed in (b)/(c)
    print("  grid refinement study (red-black SOR, optimal lambda):")
    extra = {"phi_outer": phi_outer,
             "phi_box": lambda d, T: phi_box(d, T) if d.h <= 0.5 else None,
             "phi_source": phi_source}
    st = refinement_study(hs, extra=extra, verbose=True)
    p_obs, T9_ref = extrapolate_last_three(st["T9"])
    rows = [[hlabel(h), it] + list(T.ravel()) for h, it, T in zip(st["h"], st["iters"], st["T9"])]
    rows.append(["Richardson (p_obs)", "-"] + list(T9_ref.ravel()))
    rows.append(["p_obs (last 3 grids)", "-"] + list(p_obs.ravel()))
    write_csv("a3_grid_refinement.csv", ["h", "SOR iterations"] + [f"T{p}" for p in PTS], rows)
    print(f"  observed order p (finest grids) = {np.round(p_obs.ravel(), 2)}")

    # lecture default p = 2 vs observed p
    T9_p2 = richardson(st["T9"][-2], st["T9"][-1], 2.0)

    # --- 3. how the off-grid source is modelled (h = 1)
    modes = {}
    for mode in ("exact", "snap_left", "snap_right"):
        d = discretise(prob, 1.0, mode)
        AA, bb, _ = d.to_matrix()
        modes[mode] = sample(d.to_field(gauss_elimination(AA, bb)), d)
    odd = {}
    for h in (1 / 3, 1 / 5):                                   # source again off-grid -> unequal stencil
        d = discretise(prob, h, "exact")
        odd[h] = sample(solve_grid(d, tol=1e-11).T, d)
    rows = [[m] + list(v.ravel()) + [float(np.max(np.abs(v - T9_ref)))] for m, v in modes.items()]
    rows += [[f"exact, h={hlabel(h)} (off-grid)"] + list(v.ravel()) + [float(np.max(np.abs(v - T9_ref)))]
             for h, v in odd.items()]
    write_csv("a4_source_modelling.csv", ["source treatment"] + [f"T{p}" for p in PTS] + ["max|T-T_ref|"], rows)

    # --- 4. cost of iterative solvers vs grid
    cost = {}
    for meth, hmin in (("jacobi", 1 / 16), ("gauss_seidel", 1 / 16), ("sor", min(hs))):
        hh = [h for h in hs if h <= 0.5 and h >= hmin - 1e-12]
        its = [solve_grid(discretise(prob, h), method=meth, tol=1e-8).iterations for h in hh]
        cost[{"jacobi": "Jacobi", "gauss_seidel": "Gauss-Seidel", "sor": "SOR (λ_opt)"}[meth]] = (hh, its)
    write_csv("a5_iterations_vs_grid.csv", ["method", "h", "iterations"],
              [[m, hlabel(h), it] for m, (hh, its) in cost.items() for h, it in zip(hh, its)])

    # --- 5. verification: smooth analytic solution, same code path
    hp, f = harmonic_test_problem("sinh")
    ver = []
    for h in hs[:-1]:
        d = discretise(hp, h)
        T = solve_grid(d, tol=1e-12).T
        X, Y = np.meshgrid(d.x, d.y, indexing="ij")
        ex = np.vectorize(f)(X, Y)
        ver.append((h, float(np.max(np.abs(T - ex)))))
    write_csv("a6_verification_sinh.csv", ["h", "max error"], [[hlabel(h), e] for h, e in ver])
    p_ver = np.log(ver[-2][1] / ver[-1][1]) / np.log(2)
    print(f"  verification (smooth analytic field): observed order = {p_ver:.2f} (expected 2)")

    disc_f, T_f = st["fields"][min(hs)]
    if plots:
        plotting.temperature_map(disc_f, T_f, T9_h1, T9_ref, FIG)
        plotting.solver_convergence({"Jacobi": r_j.err_history, "Gauss-Seidel": r_gs.err_history,
                                     f"SOR λ = {lam_opt:.3f}": r_sor.err_history},
                                    lam_info, lams, iters, rhos, FIG)
        plotting.iterations_vs_grid(cost, FIG)
        curves = []
        for k, p in enumerate([(2, 2), (1, 3), (3, 1)]):
            r, c = 3 - p[1], p[0] - 1
            curves.append((f"T{p}, assignment", st["h"][:-1], np.abs(st["T9"][:-1, r, c] - T9_ref[r, c])))
        curves.append(("max error, smooth analytic test", [v[0] for v in ver], [v[1] for v in ver]))
        plotting.grid_convergence(curves, FIG)
        plotting.source_modes({"exact": "unequal-spacing stencil (exact position)",
                               "snap_left": "source moved to x = 1", "snap_right": "source moved to x = 2"},
                              modes, T9_ref, FIG)
    return dict(disc=disc, T_h1=T_h1, T9_h1=T9_h1, T9_ref=T9_ref, T9_p2=T9_p2, p_obs=p_obs, st=st,
                lam=lam_info, iters={"Jacobi": r_j.iterations, "Gauss-Seidel": r_gs.iterations,
                                     "SOR": r_sor.iterations}, modes=modes, p_ver=p_ver)


# =============================================================================== PART B
def part_b(a, plots):
    print("\n=== PART (b): heat-flux density q = -k grad T ===")
    disc, T = a["disc"], a["T_h1"]
    schemes = ["central", "naive_central", "forward", "backward", "forward2", "backward2"]
    res = {s: flux_at_points(disc, T, s) for s in schemes}

    # reference gradient: central differences on the fine grids + Richardson
    st = a["st"]
    q_fine = []
    for h in st["h"][-3:]:
        d, TT = st["fields"][h]
        q_fine.append([[r["qx_k"], r["qy_k"]] for r in flux_at_points(d, TT, "central")])
    q_fine = np.array(q_fine)
    pq = observed_order(*q_fine)
    pq = np.clip(np.nan_to_num(pq, nan=1.0), 0.5, 2.0)
    q_ref = richardson(q_fine[1], q_fine[2], pq)

    rows = []
    for s in schemes:
        for k, r in enumerate(res[s]):
            err = np.hypot(r["qx_k"] - q_ref[k, 0], r["qy_k"] - q_ref[k, 1])
            rows.append([s, f"({r['x']},{r['y']})", r["T"], r["qx_k"], r["qy_k"], r["qmag_k"], r["angle_deg"], err])
    write_csv("b1_flux_h1_all_schemes.csv",
              ["scheme", "point", "T", "qx/k", "qy/k", "|q|/k", "direction_deg", "|q - q_ref|/k"], rows)
    write_csv("b2_flux_reference.csv", ["point", "qx/k ref", "qy/k ref", "|q|/k ref"],
              [[f"({x},{y})", q_ref[k, 0], q_ref[k, 1], np.hypot(*q_ref[k])]
               for k, (x, y) in enumerate(config.POINTS_9)])
    print("  q/k at h = 1 (source-aware central differences):")
    for r in res["central"]:
        print(f"    ({r['x']},{r['y']}): q = k*({r['qx_k']:8.2f}, {r['qy_k']:8.2f})   |q| = {r['qmag_k']:7.2f} k")
    mean_err = {s: np.mean([row[-1] for row in rows if row[0] == s]) for s in schemes}
    print("  mean |q - q_ref|/k per scheme: " + ", ".join(f"{s}={v:.1f}" for s, v in mean_err.items()))

    # total heat flow Phi through three surfaces
    hs = st["h"]
    ser = {"outer boundary": st["extra"]["phi_outer"], "box around source": st["extra"]["phi_box"],
           "faces of the red line": st["extra"]["phi_source"]}
    phi_h1 = phi_outer(disc, T)
    p_phi, phi_ref = extrapolate_last_three(ser["outer boundary"])
    p_src = observed_order(*ser["faces of the red line"][-3:])
    rows = [[hlabel(h)] + [ser[k][i] if ser[k][i] is not None else "" for k in ser] for i, h in enumerate(hs)]
    rows.append(["Richardson (outer, p_obs)", float(phi_ref), "", ""])
    rows.append(["p_obs", float(p_phi), "", float(p_src)])
    write_csv("b3_heat_flow_phi.csv", ["h"] + list(ser), rows)
    print(f"  Phi (per unit k): h=1 outer {phi_h1:.1f};  finest outer {ser['outer boundary'][-1]:.1f}, "
          f"box {ser['box around source'][-1]:.1f}, source faces {ser['faces of the red line'][-1]:.1f}")
    print(f"  Phi extrapolated = {phi_ref:.2f}  (p_obs outer = {p_phi:.2f}, source faces = {p_src:.2f})")

    if plots:
        disc_f, T_f = st["fields"][min(hs)]
        qx_f, qy_f = flux_field(disc_f, T_f)
        plotting.flux_map(disc_f, T_f, qx_f, qy_f, res["central"], FIG)
        plotting.phi_convergence(hs, ser, phi_ref, FIG)
    return dict(flux=res, q_ref=q_ref, phi_h1=phi_h1, phi_ref=float(phi_ref), p_phi=float(p_phi),
                p_src=float(p_src), phi_series=ser, mean_err=mean_err)


# =============================================================================== PART C
def part_c(b, plots):
    print("\n=== PART (c): heat absorbed, h = Phi * int k dt ===")
    rows, extra = integrate_k_all()
    I_main = [r[2] for r in rows if r[0].startswith("Simpson 1/3, h = 1")][0]
    vals = np.array([r[2] for r in rows])
    # "reasonable" = methods that use all five data points (excludes coarse trapezoid/Simpson, linear fit)
    best = vals[[i for i, r in enumerate(rows) if not any(s in r[0] for s in ("h = 4", "h = 2", "m = 1"))]]
    write_csv("c1_integral_k.csv", ["method", "family", "int_0^4 k dt", "comment"], rows)
    for r in rows:
        print(f"    {r[0]:48s} {r[2]:.5f}   {r[3]}")
    h_main = heat_absorbed(b["phi_ref"], I_main)
    h_coarse = heat_absorbed(b["phi_h1"], I_main)
    spread_I = (best.min(), best.max())
    write_csv("c2_heat_absorbed.csv", ["quantity", "value"], [
        ["Phi (h=1, outer boundary)", b["phi_h1"]], ["Phi (grid-converged)", b["phi_ref"]],
        ["int k dt (Simpson 1/3)", I_main], ["int k dt range (reasonable methods)", f"{spread_I[0]:.4f}-{spread_I[1]:.4f}"],
        ["h with grid-converged Phi", h_main], ["h with h=1 Phi", h_coarse]])
    print(f"  h = {b['phi_ref']:.2f} x {I_main:.4f} = {h_main:.1f}   (with the h = 1 grid: {h_coarse:.1f})")
    if plots:
        t, k = config.T_DATA, config.K_DATA
        fits = {"linear regression": extra["poly"][1], "quadratic regression": extra["poly"][2],
                "natural cubic spline": extra["spline"], "a − b·exp(−ct) (Gauss-Newton)": extra["expfit"]}
        plotting.k_fits(t, k, fits, FIG)
        plotting.k_integrals(rows, I_main, FIG)
    return dict(rows=rows, I_main=I_main, h_main=h_main, h_coarse=h_coarse, spread_I=spread_I,
                expfit=extra["expfit"])


# ============================================================================== SUMMARY
def write_summary(a, b, c, runtime):
    lam = a["lam"]
    qc = b["flux"]["central"]
    s = ["# ME5704 Project 2 - results summary", "",
         f"_Generated by `main.py` in {runtime:.1f} s. All numbers are reproducible._", "",
         "## (a) Steady-state temperature [K]", "",
         "**Grid of the assignment (h = 1, unequal-spacing stencil at the source):**", "", md_grid(a["T9_h1"]),
         f"**Grid-converged estimate** (h down to 1/{round(1 / a['st']['h'][-1])}, Richardson with observed order "
         f"p ≈ {np.nanmean(a['p_obs']):.2f}):", "", md_grid(a["T9_ref"]),
         f"* max difference h = 1 vs converged: {np.max(np.abs(a['T9_h1'] - a['T9_ref'])):.1f} K",
         f"* Richardson with the textbook p = 2 instead of p_obs changes the estimate by up to "
         f"{np.max(np.abs(a['T9_p2'] - a['T9_ref'])):.2f} K",
         f"* smooth analytic test case converges with p = {a['p_ver']:.2f} (scheme is O(h²)); the assignment "
         "converges with p ≈ 1 because of the singular end points of the line source.",
         f"* iterations (h = 1, ε_s = 1e-8 %): Jacobi {a['iters']['Jacobi']}, Gauss-Seidel "
         f"{a['iters']['Gauss-Seidel']}, SOR {a['iters']['SOR']}",
         f"* optimal λ: Young {lam['lam_young']:.4f}, golden-section {lam['lam_golden']:.4f} "
         f"({lam['evals_golden']} evaluations), parabolic interpolation {lam['lam_parabolic']:.4f} "
         f"({lam['evals_parabolic']} evaluations - slowed down by the kink of ρ(λ))", "",
         "## (b) Heat-flux density, q = k·(q_x/k, q_y/k)  (h = 1, source-aware central differences)", "",
         "| point | T [K] | q_x / k | q_y / k | abs(q) / k | direction [deg] |", "|---|---|---|---|---|---|"]
    for r in qc:
        s.append(f"| ({r['x']},{r['y']}) | {r['T']:.2f} | {r['qx_k']:.2f} | {r['qy_k']:.2f} | "
                 f"{r['qmag_k']:.2f} | {r['angle_deg']:.1f} |")
    s += ["", "Mean error vs grid-converged q (per unit k): " +
          ", ".join(f"{k} {v:.1f}" for k, v in b["mean_err"].items()), "",
          "## (c) Heat absorbed", "",
          f"* Φ (heat flow per unit k through the plate edges): h = 1 → {b['phi_h1']:.1f}, "
          f"grid-converged → **{b['phi_ref']:.1f}** (p_obs = {b['p_phi']:.2f}); "
          f"through the source faces p_obs = {b['p_src']:.2f} (tip singularity)",
          f"* ∫₀⁴ k dt: Simpson 1/3 = **{c['I_main']:.4f}**, other reasonable methods "
          f"{c['spread_I'][0]:.4f} – {c['spread_I'][1]:.4f}",
          f"* **h = Φ · ∫k dt = {c['h_main']:.1f}** (k-units × K × length × thickness); with the coarse "
          f"h = 1 grid one would get {c['h_coarse']:.1f}.",
          "* The spatial discretisation (Φ) dominates the uncertainty, not the time integration.", ""]
    path = os.path.join(ROOT, "results", "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="finest grid h = 1/32 instead of 1/64")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    hs = [h for h in config.GRID_SPACINGS if not (args.fast and h < 1 / 32)]
    t0 = time.time()
    a = part_a(hs, not args.no_plots)
    b = part_b(a, not args.no_plots)
    c = part_c(b, not args.no_plots)
    path = write_summary(a, b, c, time.time() - t0)
    print(f"\nDone in {time.time() - t0:.1f} s.  Summary: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
