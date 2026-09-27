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

def run_one(N):
    L = 30.0
    dx = L / N
    xf = np.linspace(0.0, L, N+1)
    x = 0.5*(xf[:-1]+xf[1:])
    B0f = 10.0 - xf*np.tan(np.deg2rad(10.0))
    w = np.clip(np.minimum((xf-10.0)/dx, (20.0-xf)/dx), 0.0, 1.0)
    B = hp.continuous_bed_geometry(B0f - 2.0*w)
    B0c = 10.0 - x*np.tan(np.deg2rad(10.0))
    h0 = np.maximum(B0c - hp.bed_center_values(B), 0.0)
    q = np.array([hp.pure_liquid_state(float(h), 0.0, p) for h in h0])
    m0 = np.sum(h0)*dx
    t = 0.0
    max_uphill = 0.0
    while t < 2.0 - 1e-13:
        q, d = hp.step_ssprk2(q, B, dx, t, 2.0-t, p, hp.transmissive_boundary, return_diagnostics=True)
        t += d['dt']
        h = hp.primitive(q, p)[3]
        max_uphill = max(max_uphill, float(np.sum(h[x < 10.0])*dx/m0))
    h = hp.primitive(q, p)[3]
    np.savez(OUT / f'excavation_N{N}.npz', x=x, h=h, B=hp.bed_center_values(B))
    return dict(N=N, dx=dx, max_uphill=max_uphill,
                mass_rel=float((np.sum(h)*dx-m0)/m0))

if __name__ == '__main__':
    rows = [run_one(N) for N in (75,150,300)]
    for i in range(2):
        a = np.load(OUT/f"excavation_N{rows[i]['N']}.npz")['h']
        b = np.load(OUT/f"excavation_N{rows[i+1]['N']}.npz")['h']
        br = b.reshape(len(a),2).mean(axis=1)
        rows[i]['selfdiff'] = float(np.sum(np.abs(a-br))/np.sum(np.abs(br)))
    (OUT/'excavation_zero_rheology.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
