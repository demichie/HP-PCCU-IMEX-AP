import numpy as np
import hp_pccu_2d_core as hp
p=hp.Params()
# Lake at rest on Q1 2D rough topography.
nx,ny=50,36;dx=dy=0.4
xv=np.arange(nx+1)*dx;yv=np.arange(ny+1)*dy
Xv,Yv=np.meshgrid(xv,yv)
Bv=0.25*np.sin(2*np.pi*Xv/11.0)+0.17*np.cos(2*np.pi*Yv/8.0)+0.06*np.sin(2*np.pi*(Xv+Yv)/5.0)
g=hp.geometry_from_vertices(Bv);eta0=2.0
q=hp.state_from_huv(eta0-g['B_cent'],0.0,0.0,p)
R,d=hp.spatial_residual(q,Bv,dx,dy,p,True)
assert np.max(np.abs(R)) < 2e-9, np.max(np.abs(R))
# Critical anti-striation decision.
h=np.array([0.4,0.05,0.4]);u=np.zeros(3);Bm=np.array([0.,0.,0.]);Bp=np.array([0.,0.1,0.])
hm,he=hp.reconstruct_direct_1d(h,p);hum,hue=hp.reconstruct_direct_1d(h*u,p);um,ue=hp.reconstruct_direct_1d(u,p)
out=hp.reconstruct_hp_line(h,u,Bm,Bp,hm,he,hum,hue,um,ue,np.array([0.,1.,0.]),np.array([0.,2.,0.]),p)
assert out[-1][1]==0.0
# Uniform flat 2D flow.
nx,ny=48,12;dx=dy=.25;Bv=np.zeros((ny+1,nx+1));h=np.ones((ny,nx));u=np.full_like(h,.3);v=np.zeros_like(h)
q=hp.state_from_huv(h,u,v,p);R=hp.spatial_residual(q,Bv,dx,dy,p)
assert np.max(np.abs(R))<2e-10
print('PASS: Q1 lake-at-rest, dry-safe HP gate, uniform 2-D flow')
