from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap as ap
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)

def bed(x):
 return (-x*np.tan(np.deg2rad(8.0))+0.14*np.sin(2*np.pi*x/13+0.20)+0.06*np.sin(2*np.pi*x/5.2+1.10)+0.03*np.sin(2*np.pi*x/2.4+2.00)+0.08*np.exp(-((x-31)/2.6)**2)-0.06*np.exp(-((x-42)/3.2)**2))
def pile(x):
 z=(x-16)/4; return np.where(np.abs(x-16)<4,2.5*np.cos(.5*np.pi*z)**2,0.)
def run(N,tf,outputs=()):
 p=hp.SolverParams(cfl=.24); xf=np.linspace(-15,60,N+1); dx=75/N; x=.5*(xf[:-1]+xf[1:]); B=hp.BedGeometry1D(bed(xf)); q=np.array([hp.pure_liquid_state(float(h),0,p) for h in pile(x)]); r=ap.NewtonianRheology(mu=1e5); t=0.;k=0; outs=sorted(outputs); saved={0.:q.copy()}; hist=[]
 while t<tf-1e-12:
  cap=tf-t
  if k<len(outs): cap=min(cap,outs[k]-t)
  c=ap.prepare_cache(q,B,dx,t,0,p,hp.transmissive_boundary,r,q_is_sanitized=True); sp=float(np.max(np.maximum(np.abs(c['a_minus']),np.abs(c['a_plus'])))); df=np.inf if sp<=1e-14 else .99*p.cfl*dx/sp
  q,d=ap.step_imex_euler(q,B,dx,t,cap,p,hp.transmissive_boundary,r,True);t+=d['dt'];hist.append((t,d['dt'],min(df,cap),d['w_max'],d['retries']))
  while k<len(outs) and t>=outs[k]-1e-10: saved[outs[k]]=q.copy();k+=1
 prim=hp.primitive(q,p);h=prim[3];u=prim[1];wet=np.where(h>1e-4)[0];front=float(x[wet[-1]]) if len(wet) else np.nan
 np.savez(OUT/f'rough_newtonian_N{N}_t{int(tf)}.npz',x=x,B=bed(x),h=h,u=u,**{f'h_t{int(tt)}':hp.primitive(qq,p)[3] for tt,qq in saved.items()})
 with open(OUT/f'rough_newtonian_N{N}_history.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['time','dt_ap','dt_full','wmax','retries']);w.writerows(hist)
 return dict(N=N,dx=dx,front=front,mass_rel=float((np.sum(h)-np.sum(pile(x)))/np.sum(pile(x))),mean_dt=float(np.mean([z[1] for z in hist])),median_gain=float(np.median([z[1]/z[2] for z in hist if np.isfinite(z[2]) and z[2]>0])),mean_inst_gain=float(np.mean([z[1]/z[2] for z in hist if np.isfinite(z[2]) and z[2]>0])),max_inst_gain=float(np.max([z[1]/z[2] for z in hist if np.isfinite(z[2]) and z[2]>0])))
if __name__=='__main__':
 rows=[run(80,300),run(160,600,(100,300,600)),run(320,300)]
 with open(OUT/'rough_newtonian_summary.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 for r in rows: print(r)
