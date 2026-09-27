from pathlib import Path
import sys, csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_damped_pulses import exact_gaussian, exact_tophat, square_cellavg, run, rel_l2

OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
L=40.;N=200;x=-L/2+(np.arange(N)+0.5)*(L/N);h0=1.;A=1e-3;gamma=500.;rho=1000.;mu=gamma*rho/3.;tf=10.
cases=[('Gaussian',h0+A*np.exp(-0.5*x*x),exact_gaussian(x,tf,h0,A,1.,gamma)),
       ('Square',square_cellavg(x,h0,A,2.),exact_tophat(x,tf,h0,A,2.,gamma,L=L))]
rows=[]
for name,hinit,href in cases:
    h,d=run(hinit,x,mu,tf,True,np.inf); dt0=float(np.mean(d))
    for fac in [1.,0.5,0.25]:
        if fac==1.: hh,dd=h,d
        else: hh,dd=run(hinit,x,mu,tf,True,dt0*fac)
        rows.append(dict(test=name,factor_vs_natural=fac,steps=len(dd),mean_dt=float(np.mean(dd)),L2_rel_pct=100*rel_l2(hh,href,h0)))
with open(OUT/'apoff_dt_sensitivity.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
for r in rows: print(r)
