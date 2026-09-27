from pathlib import Path
import sys, math, csv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap as ap

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

def exact_gaussian(x,t,h0,A,sigma,gamma,g=9.81):
    n=len(x); dx=x[1]-x[0]
    H=np.fft.fft(A*np.exp(-0.5*(x/sigma)**2))
    k=2*np.pi*np.fft.fftfreq(n,d=dx)
    disc=(gamma*gamma-4*g*h0*k*k).astype(complex)
    root=np.sqrt(disc); lp=(-gamma+root)/2; lm=(-gamma-root)/2; sep=lp-lm
    fac=np.ones(n,dtype=complex)
    reg=np.abs(sep)>1e-13
    fac[reg]=(-lm[reg]*np.exp(lp[reg]*t)+lp[reg]*np.exp(lm[reg]*t))/sep[reg]
    rep=~reg
    if np.any(rep): fac[rep]=(1+gamma*t/2)*np.exp(-gamma*t/2)
    fac[np.abs(k)<1e-14]=1.0
    return h0+np.fft.ifft(H*fac).real

def exact_tophat(x,t,h0,A,a,gamma,g=9.81,L=40.0,M=12000):
    dx=x[1]-x[0]
    eta=np.full_like(x,A*(2*a/L),dtype=float)
    c2=g*h0
    for n0 in range(1,M+1,1000):
        ns=np.arange(n0,min(M+1,n0+1000),dtype=float)
        k=2*np.pi*ns/L
        cn=A*2*np.sin(k*a)/(L*k)
        root=np.sqrt((gamma*gamma-4*c2*k*k).astype(complex))
        lp=(-gamma+root)/2; lm=(-gamma-root)/2; sep=lp-lm
        fac=(-lm*np.exp(lp*t)+lp*np.exp(lm*t))/sep
        sinc=np.sinc(k*dx/(2*np.pi))
        phase=np.exp(1j*np.outer(k,x))
        eta += 2*np.real(np.sum((cn*fac*sinc)[:,None]*phase,axis=0))
    return h0+eta

def square_cellavg(x,h0,A,a):
    dx=x[1]-x[0]; left=x-dx/2; right=x+dx/2
    overlap=np.maximum(0,np.minimum(right,a)-np.maximum(left,-a))
    return h0+A*overlap/dx

def make_q(h,p):
    return np.array([hp.pure_liquid_state(float(v),0.0,p) for v in h])

def run(hinit,x,mu,tfinal,force_ap_off=False,dt_cap=np.inf):
    p=hp.SolverParams(cfl=0.24,slope_correction=False,curvature_term=False)
    dx=x[1]-x[0]; B=hp.continuous_bed_geometry(np.zeros(len(x)+1)); q=make_q(hinit,p)
    rheo=ap.NewtonianRheology(mu=mu)
    old=ap.blend_weight
    if force_ap_off: ap.blend_weight=lambda chi: np.zeros_like(np.asarray(chi,float))
    t=0.; dts=[]
    try:
        while t<tfinal-1e-13:
            q,d=ap.step_imex_euler(q,B,dx,t,min(dt_cap,tfinal-t),p,hp.transmissive_boundary,rheo,True)
            t+=d['dt']; dts.append(d['dt'])
    finally:
        ap.blend_weight=old
    prim=hp.primitive(q,p)
    return prim[3],np.asarray(dts)

def rel_l2(h,href,h0):
    return np.sqrt(np.mean((h-href)**2))/np.sqrt(np.mean((href-h0)**2))

def variance(h,x,h0):
    q=np.maximum(h-h0,0.0); dx=x[1]-x[0]; m=np.sum(q)*dx; xc=np.sum(x*q)*dx/m
    return np.sum((x-xc)**2*q)*dx/m

def one_case(name,hinit,href,x,h0,mu,tfinal):
    hoff,dtoff=run(hinit,x,mu,tfinal,True,np.inf)
    small=float(np.mean(dtoff))
    hsmall,dtsmall=run(hinit,x,mu,tfinal,False,small)
    hnat,dtnat=run(hinit,x,mu,tfinal,False,np.inf)
    out=[]
    for method,h,dts in [('AP off',hoff,dtoff),('AP on, small dt',hsmall,dtsmall),('AP on, natural dt',hnat,dtnat)]:
        out.append(dict(test=name,method=method,steps=len(dts),mean_dt=float(np.mean(dts)),
                        L2_rel_pct=100*rel_l2(h,href,h0),
                        peak_rel_pct=100*abs(np.max(h-h0)-np.max(href-h0))/np.max(href-h0),
                        variance_rel_pct=100*abs(variance(h,x,h0)-variance(href,x,h0))/variance(href,x,h0)))
    return out

if __name__=='__main__':
    L=40.;N=200;x=-L/2+(np.arange(N)+0.5)*(L/N);h0=1.;A=1e-3;gamma=500.;rho=1000.;mu=gamma*rho/3.;tf=10.
    hg=h0+A*np.exp(-0.5*x*x); rg=exact_gaussian(x,tf,h0,A,1.0,gamma)
    hs=square_cellavg(x,h0,A,2.0); rs=exact_tophat(x,tf,h0,A,2.0,gamma,L=L)
    rows=one_case('Gaussian',hg,rg,x,h0,mu,tf)+one_case('Square',hs,rs,x,h0,mu,tf)
    with open(OUT/'damped_pulses_N200.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)
