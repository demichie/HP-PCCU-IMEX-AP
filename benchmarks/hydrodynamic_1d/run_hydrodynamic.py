from pathlib import Path
import sys, json
import numpy as np
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import hp_pccu_1d_core as hp

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
p = replace(hp.P0, slope_correction=False, curvature_term=False, cfl=0.24).checked()

def states(h, u=0.0):
    h = np.asarray(h)
    if np.ndim(u) == 0:
        u = np.full_like(h, float(u), dtype=float)
    return np.array([hp.pure_liquid_state(float(hh), float(uu), p) for hh, uu in zip(h, u)])

def run(q, B, dx, tend, outputs=()):
    q = hp.sanitize(q, p)
    t = 0.0
    times = sorted(set([0.0, *outputs, tend]))
    saved = {0.0: q.copy()}
    k = 1
    while t < tend - 1e-13:
        cap = tend - t
        if k < len(times):
            cap = min(cap, times[k] - t)
        q, d = hp.step_ssprk2(q, B, dx, t, cap, p, hp.transmissive_boundary, return_diagnostics=True)
        t += d['dt']
        if k < len(times) and t >= times[k] - 1e-11:
            saved[times[k]] = q.copy()
            k += 1
    return saved

def lake_at_rest():
    rows = []
    for N in [50, 100, 200, 400, 800]:
        dx = 25.0 / N
        xf = np.linspace(0.0, 25.0, N + 1)
        for sign, name in [(1.0, 'bump'), (-1.0, 'excavation')]:
            Bf = np.where(np.abs(xf - 10.0) <= 2.0,
                          sign * 0.2 * (1.0 - ((xf - 10.0) / 2.0) ** 2), 0.0)
            B = hp.continuous_bed_geometry(Bf)
            Bc = hp.bed_center_values(B)
            q = run(states(2.0 - Bc), B, dx, 2.0)[2.0]
            prim = hp.primitive(q, p)
            eta = prim[3] + Bc
            rows.append(dict(N=N, case=name,
                             eta_inf=float(np.max(np.abs(eta - 2.0))),
                             u_inf=float(np.max(np.abs(prim[1])))))
    (OUT / 'lake_at_rest.json').write_text(json.dumps(rows, indent=2))

def ritter():
    rows = []
    for N in [100, 200, 400, 800]:
        xmin, xmax, tf = -5.0, 5.0, 0.5
        dx = (xmax - xmin) / N
        x = xmin + (np.arange(N) + 0.5) * dx
        B = hp.continuous_bed_geometry(np.zeros(N + 1))
        h0 = (x < 0.0).astype(float)
        q = run(states(h0), B, dx, tf)[tf]
        prim = hp.primitive(q, p)
        h, u = prim[3], prim[1]
        hu = h * u
        c0 = np.sqrt(p.g)
        xi = x / tf
        he = np.zeros(N)
        ue = np.zeros(N)
        left = xi <= -c0
        fan = (xi > -c0) & (xi < 2.0 * c0)
        he[left] = 1.0
        ce = (2.0 * c0 - xi[fan]) / 3.0
        he[fan] = ce ** 2 / p.g
        ue[fan] = 2.0 * (c0 + xi[fan]) / 3.0
        hue = he * ue
        wet = np.where(h > 1e-3)[0]
        front = x[wet[-1]] + 0.5 * dx if len(wet) else np.nan
        rows.append(dict(N=N, dx=dx,
                         L1h=float(np.sum(np.abs(h - he)) * dx),
                         L1hu=float(np.sum(np.abs(hu - hue)) * dx),
                         front=float(front),
                         front_err=float(front - 2.0 * c0 * tf),
                         mass_rel=float((np.sum(h) - np.sum(h0)) / np.sum(h0))))
        if N == 400:
            np.savez(OUT / 'ritter_N400.npz', x=x, h=h, hu=hu, h_exact=he, hu_exact=hue)
    (OUT / 'ritter.json').write_text(json.dumps(rows, indent=2))

def rough_release():
    rows = []
    profiles = {}
    def bed(x):
        return (-x * np.tan(np.deg2rad(8.0))
                + 0.14 * np.sin(2*np.pi*x/13 + 0.20)
                + 0.06 * np.sin(2*np.pi*x/5.2 + 1.10)
                + 0.03 * np.sin(2*np.pi*x/2.4 + 2.00)
                + 0.08 * np.exp(-((x-31)/2.6)**2)
                - 0.06 * np.exp(-((x-42)/3.2)**2))
    for N in [80, 160, 320, 640]:
        L = 60.0
        dx = L / N
        xf = np.linspace(0.0, L, N + 1)
        x = 0.5 * (xf[:-1] + xf[1:])
        B = hp.continuous_bed_geometry(bed(xf))
        Bc = hp.bed_center_values(B)
        h0 = np.where(np.abs(x-16.0) < 4.0,
                      2.5*np.cos(0.5*np.pi*(x-16.0)/4.0)**2, 0.0)
        q = run(states(h0), B, dx, 2.0, outputs=[0.25, 0.5, 1.0])[2.0]
        prim = hp.primitive(q, p)
        h = prim[3]
        wet = np.where(h > 1e-4)[0]
        front = x[wet[-1]] + 0.5*dx if len(wet) else np.nan
        rows.append(dict(N=N, dx=dx, front=float(front),
                         mass_rel=float((np.sum(h)-np.sum(h0))/np.sum(h0))))
        profiles[N] = h.copy()
        np.savez(OUT / f'rough_N{N}.npz', x=x, B=Bc, h=h, eta=h+Bc)
    for i in range(len(rows)-1):
        N = rows[i]['N']
        fine = profiles[2*N].reshape(N, 2).mean(axis=1)
        rows[i]['selfdiff_to_2N'] = float(np.sum(np.abs(profiles[N]-fine))/np.sum(np.abs(fine)))
    rows[-1]['selfdiff_to_2N'] = None
    (OUT / 'rough_release.json').write_text(json.dumps(rows, indent=2))

if __name__ == '__main__':
    lake_at_rest()
    ritter()
    rough_release()
    print('Wrote hydrodynamic benchmark results to', OUT)
