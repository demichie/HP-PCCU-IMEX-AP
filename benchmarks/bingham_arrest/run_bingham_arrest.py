from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap_bingham as bg
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
def bed(x): return (-x*np.tan(np.deg2rad(8.0))+0.14*np.sin(2*np.pi*x/13+0.20)+0.06*np.sin(2*np.pi*x/5.2+1.10)+0.03*np.sin(2*np.pi*x/2.4+2.00)+0.08*np.exp(-((x-31)/2.6)**2)-0.06*np.exp(-((x-42)/3.2)**2))
def pile(x):
 z=(x-16)/4;return np.where(np.abs(x-16)<4,2.5*np.cos(.5*np.pi*z)**2,0.)
def run(N):
 p=hp.SolverParams(cfl=.24);xf=np.linspace(-15,60,N+1);dx=75/N;x=.5*(xf[:-1]+xf[1:]);B=hp.BedGeometry1D(bed(xf));q=np.array([hp.pure_liquid_state(float(h),0,p) for h in pile(x)]);r=bg.BinghamRheology(mu=1e5,tau_y=2000.,enabled=True);t=0.;hist=[]
 while t<3000-1e-12:
  q,d=bg.step_imex_euler(q,B,dx,t,3000-t,p,hp.transmissive_boundary,r,True);t+=d['dt'];prim=hp.primitive(q,p);h=prim[3];u=prim[1];wet=h>1e-4;front=float(x[np.where(wet)[0][-1]]) if np.any(wet) else np.nan;maxu=float(np.max(np.abs(u[wet]))) if np.any(wet) else 0.;hist.append((t,front,d['dt'],d.get('dt_full',np.nan),d['w_max'],maxu))
 prim=hp.primitive(q,p);h=prim[3];u=prim[1];
 with open(OUT/f'bingham_N{N}_profile.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['x','B','h','u','eta']);w.writerows(zip(x,bed(x),h,u,h+bed(x)))
 with open(OUT/f'bingham_N{N}_history.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['time','front','dt_ap','dt_full','wmax','maxu']);w.writerows(hist)
 d=np.array([z[2] for z in hist[:-1]],float);df=np.array([z[3] for z in hist[:-1]],float)
 return dict(N=N,dx=dx,front=front,maxu=maxu,mean_dt=float(d.mean()),mean_full_dt=float(np.nanmean(df)),Gmean=float(d.mean()/np.nanmean(df)))
if __name__=='__main__':
 rows=[run(N) for N in (40,80,160,320)]
 with open(OUT/'bingham_summary.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 for r in rows: print(r)
