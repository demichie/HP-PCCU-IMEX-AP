
from dataclasses import dataclass
import numpy as np
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap as base

@dataclass(frozen=True)
class BinghamRheology:
    mu: float = 1.0e5
    tau_y: float = 0.0
    enabled: bool = True

def rhs_hydro(cache,dx):
    return base.rhs_hydro(cache,dx)

def yield_gate_from_hydro(cache,dx,rheology,p):
    n=len(cache['q'])
    if (not rheology.enabled) or rheology.tau_y<=0.0:
        return np.ones(n+1),np.ones(n),np.zeros(n)
    Rh=rhs_hydro(cache,dx)
    drive=np.abs(Rh[:,1])
    phi=np.zeros(n,float)
    active=drive>rheology.tau_y
    phi[active]=1.0-rheology.tau_y/drive[active]
    face=np.empty(n+1,float)
    face[0]=phi[0]; face[-1]=phi[-1]
    if n>1:
        face[1:-1]=np.maximum(phi[:-1],phi[1:])
    return face,phi,drive

def bingham_relax(q_forced,dt,p,rheology):
    q=hp.sanitize(q_forced,p)
    prim=hp.primitive(q,p)
    h=np.asarray(prim[3]); rho=np.asarray(prim[4])
    wet=h>p.dry_h_tol
    rate=np.zeros_like(h)
    if rheology.enabled and rheology.mu>0:
        rate[wet]=3.0*rheology.mu/(rho[wet]*h[wet]**2)
    m=np.asarray(q[:,1],float)
    thr=float(dt)*float(rheology.tau_y)
    shrunk=np.sign(m)*np.maximum(np.abs(m)-thr,0.0)
    q[:,1]=np.where(wet,shrunk/(1.0+dt*rate),0.0)
    yielded=wet & (np.abs(m)>thr)
    return hp.sanitize(q,p),rate,yielded

def cfl_dt(q,B,dx,t,dt_cap,p,boundary,rheology):
    nr=base.NewtonianRheology(mu=rheology.mu,enabled=rheology.enabled)
    cache=base.prepare_cache(q,B,dx,t,0.0,p,boundary,nr,q_is_sanitized=True)
    uface=np.maximum(np.abs(cache['data']['uL_face']),np.abs(cache['data']['uR_face']))
    speed_hp=np.maximum(np.abs(cache['a_minus']),np.abs(cache['a_plus']))
    speed_ap=uface+2.0*cache['nu_ap']
    speed_face=(1.0-cache['w'])*speed_hp+cache['w']*speed_ap
    speed=float(np.max(speed_face))
    if speed<=1e-14: return float(dt_cap),cache,speed
    return min(float(dt_cap),0.99*p.cfl*float(dx)/speed),cache,speed

def step_imex_euler(q,B,dx,t,dt_cap,p,boundary,rheology=BinghamRheology(),return_diagnostics=False,max_retries=20):
    qn=hp.sanitize(q,p)
    dt,cache0,speed=cfl_dt(qn,B,dx,t,dt_cap,p,boundary,rheology)
    nr=base.NewtonianRheology(mu=rheology.mu,enabled=rheology.enabled)
    retries=0
    while True:
        cache_force=base.prepare_cache(qn,B,dx,t,dt,p,boundary,nr,q_is_sanitized=True)
        gate_face,gate_cell,drive=yield_gate_from_hydro(cache_force,dx,rheology,p)
        cache_n=base.prepare_cache(qn,B,dx,t,dt,p,boundary,nr,q_is_sanitized=True,diffusion_gate=gate_face)
        Radv=base.rhs_adv(cache_n,dx)
        qstar_raw=qn+dt*Radv
        hraw=hp.raw_thickness(qstar_raw,p)
        tol=5e-13*max(1.0,float(np.max(cache_n['data']['prim_q'][3])))
        if float(np.min(hraw))>=-tol:
            qstar=hp.sanitize(qstar_raw,p)
            cache_s=base.prepare_cache(qstar,B,dx,t+dt,dt,p,boundary,nr,q_is_sanitized=True)
            Rh=base.rhs_hydro(cache_s,dx)
            qforced=qstar.copy(); qforced[:,1]+=dt*Rh[:,1]
            qnew,rate,yielded=bingham_relax(qforced,dt,p,rheology)
            qnew[:,0]=qstar[:,0]; qnew[:,2]=qstar[:,2]
            qnew=hp.sanitize(qnew,p)
            speed_full=float(np.max(np.maximum(np.abs(cache0['a_minus']),np.abs(cache0['a_plus']))))
            dt_full=(float(dt_cap) if speed_full<=1e-14 else min(float(dt_cap),0.99*p.cfl*float(dx)/speed_full))
            diag=dict(dt=dt,dt_full=dt_full,retries=retries,speed_max=speed,
                      w_max=float(np.max(cache_n['w'])),
                      yielded_cells=int(np.count_nonzero(yielded)),
                      wet_cells=int(np.count_nonzero(hp.primitive(qnew,p)[3]>p.dry_h_tol)),
                      gate_active_cells=int(np.count_nonzero(gate_cell>0)),
                      max_drive_stress=float(np.max(drive)))
            return (qnew,diag) if return_diagnostics else qnew
        retries+=1
        if retries>max_retries: raise RuntimeError("positivity retry failed")
        dt*=0.5
