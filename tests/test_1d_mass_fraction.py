import numpy as np
import hp_pccu_1d_core as hp
p=hp.SolverParams().checked()
for a in (0.0,0.1,0.35,0.8,1.0):
    q=hp.state_from_h_alpha_u(1.7,a,0.4,p)
    y=float(hp.primitive(q,p)[6])
    qy=hp.state_from_h_massfrac_hu(1.7,y,1.7*0.4,p)
    assert np.allclose(q,qy,rtol=2e-14,atol=2e-14)
x=np.linspace(0,1,32);y=0.05+0.7*x;h=1+0.1*np.sin(2*np.pi*x);u=0.2*np.ones_like(x)
q=np.stack([hp.state_from_h_massfrac_hu(hh,yy,hh*uu,p) for hh,yy,uu in zip(h,y,u)])
qL,qR=hp.reconstruct_state(q,p)
yL=hp.primitive(qL,p)[6];yR=hp.primitive(qR,p)[6];refL,refR=hp.reconstruct_array(y,p)
assert np.allclose(yL,refL,rtol=2e-13,atol=2e-13)
assert np.allclose(yR,refR,rtol=2e-13,atol=2e-13)
print('PASS: 1-D mass-fraction reconstruction')
