from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap as ap
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)

def setup(N):
 xmin,xmax=0.,30.;dx=(xmax-xmin)/N;xf=np.linspace(xmin,xmax,N+1);x=.5*(xf[:-1]+xf[1:]);B0f=10.-xf*np.tan(np.deg2rad(10.));w=np.clip(np.minimum((xf-10.)/dx,(20.-xf)/dx),0.,1.);B=hp.continuous_bed_geometry(B0f-2*w);B0c=10.-x*np.tan(np.deg2rad(10.));h0=np.maximum(B0c-hp.bed_center_values(B),0.);p=hp.SolverParams(cfl=.24,theta=1.3,hp_dynamic_residual_threshold=.90).checked();q=np.array([hp.pure_liquid_state(float(h),0,p) for h in h0]);return dx,x,B,q,p

def full_dt(q,B,dx,p,r):
 c=ap.prepare_cache(q,B,dx,0,0,p,hp.transmissive_boundary,r,q_is_sanitized=True);s=float(np.max(np.maximum(np.abs(c['a_minus']),np.abs(c['a_plus']))));return np.inf if s<=1e-14 else .99*p.cfl*dx/s

def run(N,mu,tend=2.):
 dx,x,B,q,p=setup(N);r=ap.NewtonianRheology(mu);m0=np.sum(q[:,0])*dx;t=0.;dts=[];dfs=[]
 while t<tend-1e-13:
  cap=tend-t;df=full_dt(q,B,dx,p,r);q,d=ap.step_imex_euler(q,B,dx,t,cap,p,hp.transmissive_boundary,r,True);t+=d['dt'];dts.append(d['dt']);dfs.append(min(df,cap))
 h=hp.primitive(q,p)[3];u=hp.primitive(q,p)[1]
 np.savez(OUT/f'excavation_N{N}_mu{mu:g}.npz',x=x,h=h,u=u,B=hp.bed_center_values(B))
 return dict(N=N,mu=mu,dx=dx,steps=len(dts),mean_dt=float(np.mean(dts)),mean_full_dt=float(np.mean(dfs)),Gmean=float(np.mean(dts)/np.mean(dfs)),mass_rel=float((np.sum(q[:,0])*dx-m0)/m0))
if __name__=='__main__':
 rows=[]
 for mu in (0.,1e-3,1e3,1e5): rows.append(run(150,mu))
 for N in (75,150,300): rows.append(run(N,1e5))
 with open(OUT/'excavation_1d_summary.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 for r in rows: print(r)
