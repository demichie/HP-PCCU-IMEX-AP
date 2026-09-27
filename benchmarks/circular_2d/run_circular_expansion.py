from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_2d_core as hp
import hp_pccu_2d_imex_ap as ia
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
p=hp.Params(cfl=.24)

def disk_cell_average(xc,yc,dx,sub=16):
 offs=((np.arange(sub)+.5)/sub-.5)*dx;Xc,Yc=np.meshgrid(xc,yc);acc=np.zeros_like(Xc)
 for ox in offs:
  for oy in offs: acc += (((Xc+ox)**2+(Yc+oy)**2)<1.).astype(float)
 return acc/(sub*sub)

def contour_radius(h,xc,yc,level=1e-2,nang=360):
 # Bilinear sampling along rays, returning outermost threshold crossing.
 th=np.linspace(0,2*np.pi,nang,endpoint=False);rvals=np.zeros(nang);dx=xc[1]-xc[0];dy=yc[1]-yc[0]
 for j,a in enumerate(th):
  rs=np.linspace(0,3.0,1201);xs=rs*np.cos(a);ys=rs*np.sin(a);ix=(xs-xc[0])/dx;iy=(ys-yc[0])/dy;i0=np.floor(ix).astype(int);j0=np.floor(iy).astype(int);ok=(i0>=0)&(i0<len(xc)-1)&(j0>=0)&(j0<len(yc)-1);vals=np.zeros_like(rs);ii=i0[ok];jj=j0[ok];tx=ix[ok]-ii;ty=iy[ok]-jj;vals[ok]=(1-tx)*(1-ty)*h[jj,ii]+tx*(1-ty)*h[jj,ii+1]+(1-tx)*ty*h[jj+1,ii]+tx*ty*h[jj+1,ii+1];ind=np.where(vals>=level)[0];rvals[j]=rs[ind[-1]] if len(ind) else 0.
 return th,rvals

def run(N,mu):
 L=8.;dx=L/N;dy=dx;xv=np.linspace(-4,4,N+1);yv=np.linspace(-4,4,N+1);xc=.5*(xv[:-1]+xv[1:]);yc=.5*(yv[:-1]+yv[1:]);h0=disk_cell_average(xc,yc,dx,16);q0=hp.state_from_huv(h0,0.,0.,p);Bv=np.zeros((N+1,N+1));sn,st=ia.run(q0,Bv,dx,dy,.1,ia.NewtonianRheology(mu),p,dt_cap=5.);q=sn[-1][1];h,u,v=hp.primitive(q,p);th,R=contour_radius(h,xc,yc);mean=float(R.mean());E=float(np.sqrt(np.mean((R-mean)**2))/mean);np.savez(OUT/f'circular_N{N}_mu{mu:g}.npz',x=xc,y=yc,h=h,theta=th,R=R);return dict(N=N,mu=mu,mean_radius=mean,E_iso=E,steps=st['steps'],mean_dt=st['mean_dt'])
if __name__=='__main__':
 rows=[]
 for N in (40,80,120):
  for mu in (0.,1e5): rows.append(run(N,mu))
 with open(OUT/'circular_summary.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 for r in rows: print(r)
