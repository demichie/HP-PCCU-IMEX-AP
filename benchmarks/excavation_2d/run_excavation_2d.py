from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_2d_core as hp
import hp_pccu_2d_imex_ap as ia
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
p=hp.Params(cfl=.24)

def setup(dx):
 dy=dx;xv=np.arange(0.,30.+.5*dx,dx);yv=np.arange(-10.,10.+.5*dy,dy);x=.5*(xv[:-1]+xv[1:]);y=.5*(yv[:-1]+yv[1:]);X,Y=np.meshgrid(x,y);Bv=hp.inclined_excavation_vertices(xv,yv,10,10,20,5,2,10,dx);q0=hp.filled_excavation_ic_from_geometry(Bv,p,10,10,x);h0=hp.primitive(q0,p)[0];V0=float(np.sum(h0)*dx*dy);return dy,x,y,X,Y,Bv,q0,V0

def metrics(q,X,Y,dx,dy,V0):
 h,u,v=hp.primitive(q,p);m=h*dx*dy;masks={'uphill':X<10,'downhill':X>=20,'lateral_up':(X>=10)&(X<15)&(np.abs(Y)>=5),'lateral_low':(X>=15)&(X<20)&(np.abs(Y)>=5)};o={k:float(m[z].sum()/V0) for k,z in masks.items()};o['mass_ratio']=float(m.sum()/V0);o['max_speed']=float(np.max(np.sqrt(u*u+v*v)));return o

def run(dx,mu,tend=.5):
 dy,x,y,X,Y,Bv,q0,V0=setup(dx);sn,st=ia.run(q0,Bv,dx,dy,tend,ia.NewtonianRheology(mu),p,dt_cap=5.);q=sn[-1][1];o=metrics(q,X,Y,dx,dy,V0);h,u,v=hp.primitive(q,p);np.savez(OUT/f'excavation_2d_dx{dx:g}_mu{mu:g}.npz',x=x,y=y,h=h,u=u,v=v);return {"dx":dx,"mu":mu,**o,**st}
if __name__=='__main__':
 rows=[]
 for dx in (.4,.2):
  for mu in (0.,1e-3,1e3,1e5): rows.append(run(dx,mu,.5))
 with open(OUT/'excavation_2d_summary.csv','w',newline='') as f:
  keys=sorted(set().union(*[r.keys() for r in rows]));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 for r in rows: print(r)
