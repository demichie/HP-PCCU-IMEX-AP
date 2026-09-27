#!/usr/bin/env python3
"""Reproduce the four fully wet SWASHES smooth-bump steady-flow tests.

The driver uses the current 1-D HP-PCCU mass-fraction core in ``src/``.
For each case and grid, the numerical state is initialized from the analytical
steady solution and advanced for a short drift test (default 0.2 s). This is
the protocol used for the smooth-bump refinement data in the manuscript work.

No figures are generated. Machine-readable CSV/NPZ outputs are written under
``benchmarks/smooth_bump/results/``.
"""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hp_pccu_1d_core as hp

OUT = Path(__file__).resolve().parent / "results"

CASES = [
    dict(name="subcritical", q=4.42, h_left=None, h_right=2.0),
    dict(name="supercritical", q=10.0, h_left=1.0, h_right=None),
    dict(name="transcritical_no_shock", q=1.53, h_left=None, h_right=None),
    dict(name="transcritical_with_shock", q=0.18, h_left=None, h_right=0.33),
]


def bump(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x - 10.0) <= 2.0,
                    0.2 - 0.05 * (x - 10.0) ** 2, 0.0)


def critical_depth(discharge, g):
    return (float(discharge) ** 2 / g) ** (1.0 / 3.0)


def specific_energy(h, discharge, g):
    h = np.asarray(h, dtype=float)
    return h + float(discharge) ** 2 / (2.0 * g * h ** 2)


def specific_energy_root(E, discharge, branch, g, tol=1.0e-13):
    """Positive root of h + q^2/(2 g h^2) = E on a selected branch."""
    q = float(discharge)
    hc = critical_depth(q, g)
    Emin = 1.5 * hc
    scale = max(1.0, abs(E), abs(Emin))
    if E < Emin - 64.0 * np.finfo(float).eps * scale:
        return np.nan
    if abs(E - Emin) <= 64.0 * np.finfo(float).eps * scale:
        return hc

    def f(h):
        return h + q * q / (2.0 * g * h * h) - E

    if branch == "supercritical":
        lo = max(1.0e-14, 1.0e-12 * hc)
        hi = hc
    elif branch == "subcritical":
        lo = hc
        hi = max(2.0 * hc, 2.0 * E, 1.0)
        while f(hi) <= 0.0:
            hi *= 2.0
    else:
        raise ValueError("branch must be 'subcritical' or 'supercritical'")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= tol * max(1.0, abs(E)):
            return mid
        if branch == "supercritical":
            if fm > 0.0:
                lo = mid
            else:
                hi = mid
        else:
            if fm < 0.0:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


def energy_profile(total_head, discharge, B, branches, g):
    B = np.asarray(B, dtype=float)
    if isinstance(branches, str):
        branches = np.full(B.shape, branches, dtype=object)
    else:
        branches = np.asarray(branches, dtype=object)
    out = np.empty_like(B)
    for i, (Bi, branch) in enumerate(zip(B, branches)):
        out[i] = specific_energy_root(total_head - Bi, discharge, str(branch), g)
    return out


def momentum_function(h, discharge, g):
    return 0.5 * h * h + float(discharge) ** 2 / (g * h)


def analytical_solution(case, x, g):
    x = np.asarray(x, dtype=float)
    B = bump(x)
    q = float(case["q"])
    name = case["name"]
    xshock = np.nan

    if name == "subcritical":
        H = float(specific_energy(case["h_right"], q, g))
        h = energy_profile(H, q, B, "subcritical", g)

    elif name == "supercritical":
        H = float(specific_energy(case["h_left"], q, g))
        h = energy_profile(H, q, B, "supercritical", g)

    elif name == "transcritical_no_shock":
        H = 0.2 + 1.5 * critical_depth(q, g)
        branches = np.where(x < 10.0, "subcritical", "supercritical")
        h = energy_profile(H, q, B, branches, g)
        h[np.isclose(x, 10.0, atol=1.0e-13, rtol=0.0)] = critical_depth(q, g)

    elif name == "transcritical_with_shock":
        Hup = 0.2 + 1.5 * critical_depth(q, g)
        Hdown = float(specific_energy(case["h_right"], q, g))

        def residual(xs):
            Bs = float(bump(np.array([xs]))[0])
            h1 = specific_energy_root(Hup - Bs, q, "supercritical", g)
            h2 = specific_energy_root(Hdown - Bs, q, "subcritical", g)
            return momentum_function(h1, q, g) - momentum_function(h2, q, g)

        # The physical hydraulic jump is the non-trivial root downstream of
        # the crest. Starting the scan slightly to the right avoids the
        # critical-point degeneracy at x=10.
        scan = np.linspace(10.001, 12.0, 4000)
        vals = np.array([residual(xx) for xx in scan])
        changes = np.where(vals[:-1] * vals[1:] < 0.0)[0]
        if len(changes) == 0:
            raise RuntimeError("Could not bracket the stationary hydraulic jump")
        j = int(changes[-1])
        a, b = float(scan[j]), float(scan[j + 1])
        fa = residual(a)
        for _ in range(120):
            m = 0.5 * (a + b)
            fm = residual(m)
            if abs(fm) < 1.0e-14 or abs(b - a) < 1.0e-13:
                a = b = m
                break
            if fa * fm <= 0.0:
                b = m
            else:
                a, fa = m, fm
        xshock = 0.5 * (a + b)

        upstream = x < xshock
        h = np.empty_like(x)
        branches_up = np.where(x[upstream] < 10.0,
                               "subcritical", "supercritical")
        h[upstream] = energy_profile(Hup, q, B[upstream], branches_up, g)
        h[~upstream] = energy_profile(Hdown, q, B[~upstream], "subcritical", g)
    else:
        raise ValueError(name)

    u = q / h
    return h, u, B, xshock


def boundary_for(case):
    q_in = float(case["q"])
    h_left = case["h_left"]
    h_right = case["h_right"]

    def boundary(side, t, q_edge, p):
        vals = hp.primitive(q_edge[None, :], p)
        u_edge = float(vals[1][0])
        h_edge = float(vals[3][0])
        if side == "left":
            h = float(h_left) if h_left is not None else h_edge
            return hp.pure_liquid_state(h, q_in / max(h, 1.0e-14), p)
        if h_right is None:
            return q_edge.copy()
        qvol = h_edge * u_edge
        h = float(h_right)
        return hp.pure_liquid_state(h, qvol / max(h, 1.0e-14), p)

    return boundary


def integrate(q, B, dx, tfinal, p, boundary):
    t = 0.0
    nstep = 0
    retries = 0
    while t < tfinal - 1.0e-14:
        q, d = hp.step_ssprk2(q, B, dx, t, tfinal - t, p, boundary,
                              return_diagnostics=True)
        t += d["dt"]
        nstep += 1
        retries += d["retries"]
    return q, nstep, retries


def run_one(case, N, tfinal, p):
    dx = 25.0 / N
    xf = np.linspace(0.0, 25.0, N + 1)
    x = 0.5 * (xf[:-1] + xf[1:])
    Bgeom = hp.continuous_bed_geometry(bump(xf))
    Bc = hp.bed_center_values(Bgeom)
    h_exact, u_exact, B_exact, xshock = analytical_solution(case, x, p.g)

    q0 = np.array([hp.pure_liquid_state(h, u, p)
                   for h, u in zip(h_exact, u_exact)])
    qf, nstep, retries = integrate(q0, Bgeom, dx, tfinal, p,
                                   boundary_for(case))
    prim = hp.primitive(qf, p)
    h = prim[3]
    u = prim[1]
    hu = h * u
    q0vol = float(case["q"])

    return dict(
        case=case["name"], N=N, dx=dx, test_time=tfinal,
        L1_h=float(np.sum(np.abs(h - h_exact)) * dx),
        Linf_h=float(np.max(np.abs(h - h_exact))),
        L1_hu=float(np.sum(np.abs(hu - q0vol)) * dx),
        Linf_hu=float(np.max(np.abs(hu - q0vol))),
        steps=nstep, positivity_retries=retries,
        xshock_exact=float(xshock) if np.isfinite(xshock) else np.nan,
        x=x, B_geometry=Bc, B_exact=B_exact,
        h=h, u=u, h_exact=h_exact, u_exact=u_exact,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", type=int, default=[40, 80, 160, 320])
    ap.add_argument("--test-time", type=float, default=0.2,
                    help="short drift time from the exact steady state [s]")
    ap.add_argument("--quick", action="store_true",
                    help="run N=80 only")
    args = ap.parse_args()
    grids = [80] if args.quick else args.grids

    p = replace(hp.P0, slope_correction=False, curvature_term=False,
                cfl=0.24).checked()
    OUT.mkdir(exist_ok=True)

    rows = []
    for case in CASES:
        for N in grids:
            r = run_one(case, N, args.test_time, p)
            rows.append({k: r[k] for k in (
                "case", "N", "dx", "test_time", "L1_h", "Linf_h",
                "L1_hu", "Linf_hu", "steps", "positivity_retries",
                "xshock_exact")})
            np.savez(OUT / f"{case['name']}_N{N}.npz",
                     x=r["x"], B_geometry=r["B_geometry"], B_exact=r["B_exact"],
                     h=r["h"], u=r["u"], h_exact=r["h_exact"],
                     u_exact=r["u_exact"])
            print(f"{case['name']:30s} N={N:4d}  L1(h)={r['L1_h']:.8e}  "
                  f"steps={r['steps']:4d}")

    fields = ["case", "N", "dx", "test_time", "L1_h", "Linf_h",
              "L1_hu", "Linf_hu", "steps", "positivity_retries", "xshock_exact"]
    with (OUT / "smooth_bump_errors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print("Wrote", OUT / "smooth_bump_errors.csv")


if __name__ == "__main__":
    main()
