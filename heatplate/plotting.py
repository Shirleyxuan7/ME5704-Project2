"""Figures for the report (matplotlib, saved as PNG)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

# categorical colours, always assigned in this order (validated colour-blind-safe set)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SOURCE_RED = "#d03b3b"
CMAP = "Oranges"           # sequential, single hue (magnitude = temperature)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plate(ax):
    ax.plot([config.SOURCE_X] * 2, config.SOURCE_Y, color=SOURCE_RED, lw=3, solid_capstyle="round",
            label="heat source 350 K")
    ax.set_xlim(0, config.L); ax.set_ylim(0, config.L); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(False)


def temperature_map(disc_f, T_f, T9_coarse, T9_ref, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    X, Y = np.meshgrid(disc_f.x, disc_f.y, indexing="ij")
    cf = ax.contourf(X, Y, T_f, levels=np.linspace(200, 350, 31), cmap=CMAP)
    cs = ax.contour(X, Y, T_f, levels=np.arange(210, 350, 10), colors="#52514e", linewidths=0.5)
    ax.clabel(cs, levels=cs.levels[::2], fontsize=7, fmt="%d")
    _plate(ax)
    for r, yy in enumerate((3, 2, 1)):
        for c, xx in enumerate((1, 2, 3)):
            ax.plot(xx, yy, "o", color=INK, ms=4)
            ax.annotate(f"{T9_coarse[r, c]:.1f}\n({T9_ref[r, c]:.1f})", (xx, yy), xytext=(4, 4),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    fig.colorbar(cf, ax=ax, label="T [K]", ticks=range(200, 351, 25))
    ax.set_title("Steady-state temperature (fine grid h = 1/64)\n"
                 "labels: h = 1 result  (grid-converged estimate)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, outdir, "fig1_temperature_field.png")


def solver_convergence(histories, lam_info, lams, iters, rhos, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axs[0]
    for k, (name, hist) in enumerate(histories.items()):
        ax.semilogy(range(1, len(hist) + 1), hist, color=C[k], marker=MARKERS[k], ms=3, label=name)
    ax.set_xlabel("iteration"); ax.set_ylabel("max approx. relative error  ε_a [%]")
    ax.set_title("(a) Convergence history, h = 1"); ax.legend(fontsize=8)

    ax = axs[1]
    ax.plot(lams, rhos, color=C[0], label="ρ(λ) of SOR iteration matrix")
    ax.axvline(lam_info["lam_young"], color=INK2, ls="--", lw=1)
    ax.annotate(f"Young λ_opt = {lam_info['lam_young']:.4f}", (lam_info["lam_young"], 0.45),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2)
    hg = lam_info["hist_golden"]
    ax.plot(hg, [np.interp(h, lams, rhos) for h in hg], "o", color=C[1], ms=4, label="golden-section iterates")
    hp = lam_info["hist_parabolic"][:15]
    ax.plot(hp, [np.interp(h, lams, rhos) for h in hp], "s", mfc="none", color=C[2], ms=6,
            label="parabolic-interp. iterates")
    ax.set_xlabel("relaxation factor λ"); ax.set_ylabel("spectral radius ρ")
    ax.set_title("(b) 1-D optimisation of λ"); ax.legend(fontsize=8, loc="upper left")

    ax = axs[2]
    ax.plot(lams, iters, color=C[0], drawstyle="steps-mid")
    ax.axvline(lam_info["lam_young"], color=INK2, ls="--", lw=1)
    ax.set_xlabel("relaxation factor λ"); ax.set_ylabel("iterations to ε_a < 1e-8 %")
    ax.set_title("(c) Iterations vs λ, h = 1")
    fig.tight_layout()
    return _save(fig, outdir, "fig2_solver_convergence.png")


def iterations_vs_grid(table, outdir):
    """table: {method: (hs, iterations)}"""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for k, (name, (hs, its)) in enumerate(table.items()):
        n = [4 / h for h in hs]
        ax.loglog(n, its, color=C[k], marker=MARKERS[k], label=name)
        ax.annotate(name, (n[-1], its[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.set_xlabel("intervals per side  n = 4/h"); ax.set_ylabel("iterations to converge")
    ax.set_title("Cost of iterative solvers as the grid is refined")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "fig3_iterations_vs_grid.png")


def grid_convergence(curves, outdir):
    """curves: list of (label, hs, errors)"""
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    hs_all = np.array(sorted({h for _, hs, _ in curves for h in hs}))
    for k, (label, hs, err) in enumerate(curves):
        ax.loglog(hs, err, color=C[k], marker=MARKERS[k], label=label)
    h0 = hs_all.max(); ref = np.array([hs_all.min(), h0])
    ax.loglog(ref, 20 * (ref / h0) ** 1, color=INK2, ls=":", lw=1)
    ax.loglog(ref, 2 * (ref / h0) ** 2, color=INK2, ls="--", lw=1)
    ax.annotate("slope 1 (reference)", (ref[0], 20 * (ref[0] / h0)), fontsize=8, color=INK2, xytext=(4, -14), textcoords="offset points")
    ax.annotate("slope 2 (reference)", (ref[0], 2 * (ref[0] / h0) ** 2), fontsize=8, color=INK2, xytext=(14, -4), textcoords="offset points")
    ax.set_xlabel("grid spacing h"); ax.set_ylabel("|error|")
    ax.set_title("Grid convergence: assignment problem vs smooth analytic test")
    ax.legend(fontsize=7.5, loc="lower right")
    return _save(fig, outdir, "fig4_grid_convergence.png")


def source_modes(labels, values, ref, outdir):
    """values: {mode: 3x3}, ref: 3x3"""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    pts = [f"({x},{y})" for y in (3, 2, 1) for x in (1, 2, 3)]
    xpos = np.arange(9)
    w = 0.8 / len(values)
    for k, (mode, v) in enumerate(values.items()):
        ax.bar(xpos + (k - (len(values) - 1) / 2) * w, (v - ref).ravel(), width=w * 0.92,
               color=C[k], label=labels[mode])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(pts)
    ax.set_ylabel("T(h=1) − T_converged  [K]")
    ax.set_title("Effect of how the off-grid source is modelled (h = 1)")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "fig5_source_modelling.png")


def flux_map(disc_f, T_f, qx_f, qy_f, rows_h1, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    X, Y = np.meshgrid(disc_f.x, disc_f.y, indexing="ij")
    ax.contourf(X, Y, T_f, levels=np.linspace(200, 350, 31), cmap=CMAP, alpha=0.55)
    s = max(1, disc_f.n // 16)
    mag = np.hypot(qx_f, qy_f)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = qx_f / mag, qy_f / mag
    ax.quiver(X[::s, ::s], Y[::s, ::s], ux[::s, ::s], uy[::s, ::s], color=INK2, scale=40, width=0.003,
              label="direction of q (fine grid)")
    qx = np.array([r["qx_k"] for r in rows_h1]); qy = np.array([r["qy_k"] for r in rows_h1])
    xs = np.array([r["x"] for r in rows_h1]); ys = np.array([r["y"] for r in rows_h1])
    ax.quiver(xs, ys, qx, qy, color=C[0], angles="xy", scale_units="xy", scale=150, width=0.008,
              label="q/k at the 9 nodes (h = 1)")
    for r in rows_h1:
        ax.annotate(f"{r['qmag_k']:.0f}", (r["x"], r["y"]), xytext=(-14, -12), textcoords="offset points",
                    fontsize=7.5, color=C[0])
    _plate(ax)
    ax.set_title("Heat-flux density  q = −k∇T  (numbers: |q|/k at h = 1)")
    ax.legend(loc="lower right", fontsize=7.5)
    return _save(fig, outdir, "fig6_heat_flux.png")


def phi_convergence(hs, series, phi_ref, outdir):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k, (name, vals) in enumerate(series.items()):
        hh = [h for h, v in zip(hs, vals) if v is not None]
        vv = [v for v in vals if v is not None]
        ax.semilogx(hh, vv, color=C[k], marker=MARKERS[k], label=name)
    ax.axhline(phi_ref, color=INK2, ls="--", lw=1)
    ax.annotate(f"extrapolated Φ = {phi_ref:.1f}", (max(hs), phi_ref), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing h  (finer →)"); ax.set_ylabel("Φ = heat flow per unit k  [K]")
    ax.set_title("Energy balance: heat leaving the source through different surfaces")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "fig7_heat_flow_phi.png")


def k_fits(t, k, fits, outdir):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    tt = np.linspace(0, 4, 200)
    for idx, (name, f) in enumerate(fits.items()):
        ax.plot(tt, f(tt), color=C[idx], label=name, lw=1.6)
    ax.plot(t, k, "o", color=INK, ms=7, label="data", zorder=5)
    ax.set_xlabel("t"); ax.set_ylabel("k"); ax.set_ylim(0, 0.5)
    ax.set_title("Thermal conductivity data and fitted models")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, outdir, "fig8_k_fits.png")


def k_integrals(rows, main_value, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    names = [r[0] for r in rows][::-1]
    vals = np.array([r[2] for r in rows])[::-1]
    fam = [r[1] for r in rows][::-1]
    colors = [C[0] if f == "data" else C[1] for f in fam]
    ax.barh(names, vals, color=colors, height=0.6)
    for y, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.axvline(main_value, color=INK2, ls="--", lw=1)
    ax.set_xlim(0.95, 1.33)
    ax.set_xlabel("∫₀⁴ k dt")
    ax.set_title("∫k dt by method  (blue: on data, orange: fit then integrate)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    return _save(fig, outdir, "fig9_k_integrals.png")
