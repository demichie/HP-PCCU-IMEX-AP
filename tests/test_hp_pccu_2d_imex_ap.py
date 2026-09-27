import numpy as np
import hp_pccu_2d_core as hp
import hp_pccu_2d_imex_ap as ia

p=hp.Params(cfl=0.24)
# 1) Split sums exactly to original spatial residual when mu=0.
nx,ny=30,20; dx=dy=.4
xv=np.arange(nx+1)*dx; yv=np.arange(ny+1)*dy
Xv,Yv=np.meshgrid(xv,yv)
Bv=.17*np.sin(2*np.pi*Xv/7)+.11*np.cos(2*np.pi*Yv/5)
g=hp.geometry_from_vertices(Bv)
h=1.4-g['B_cent']+.03*np.exp(-((0.5*(xv[:-1]+xv[1:])[None,:]-5)/1.2)**2)
q=hp.state_from_huv(h,.05,0.,p)
R=hp.spatial_residual(q,Bv,dx,dy,p)
Ra,Rh=ia.split_residuals(q,Bv,dx,dy,ia.NewtonianRheology(0.),p)
assert np.max(np.abs(R-(Ra+Rh))) < 2e-9

# 2) Strong-rheology lake at rest stays exactly at rest under split IMEX.
eta0=2.0
q0=hp.state_from_huv(eta0-g['B_cent'],0.,0.,p)
sn,st=ia.run(q0,Bv,dx,dy,5.0,ia.NewtonianRheology(1e6),p,dt_cap=1.0)
h,u,v=hp.primitive(sn[-1][1],p)
assert np.max(np.abs(h+g['B_cent']-eta0)) < 2e-13
assert np.max(np.abs(u)) == 0.0 and np.max(np.abs(v)) == 0.0

# 3) Tiny viscosity across a wet/dry excavation does not trigger positivity retries.
dx=dy=2.0
xv=np.arange(0.,30.+1e-12,dx); yv=np.arange(-10.,10.+1e-12,dy)
xc=.5*(xv[:-1]+xv[1:])
Bv=hp.inclined_excavation_vertices(xv,yv,10,10,20,5,2,10,dx)
q0=hp.filled_excavation_ic_from_geometry(Bv,p,10,10,xc)
sn,st=ia.run(q0,Bv,dx,dy,.5,ia.NewtonianRheology(1e-3),p,dt_cap=5.)
assert st['retries']==0
assert abs(np.sum(hp.primitive(sn[-1][1],p)[0])-np.sum(hp.primitive(q0,p)[0]))/np.sum(hp.primitive(q0,p)[0]) < 1e-9
print('PASS: split equivalence, strong-rheology lake at rest, tiny-mu wet/dry robustness')
