
from dataclasses import dataclass
import numpy as np
import hp_pccu_1d_core as hp

@dataclass(frozen=True)
class NewtonianRheology:
    mu: float = 1.0e5
    enabled: bool = True

def relaxation_rate_from_primitive(prim,rheology,p):
    h=np.asarray(prim[3],float); rho=np.asarray(prim[4],float)
    rate=np.zeros_like(h)
    wet=h>p.dry_h_tol
    if rheology.enabled and rheology.mu>0:
        rate[wet]=3.0*rheology.mu/(rho[wet]*h[wet]**2)
    return rate

def chi_from_primitive(prim,dx,rheology,p):
    h=np.asarray(prim[3],float); gprime=np.asarray(prim[7],float)
    c=np.sqrt(np.maximum(gprime*h,0.0))
    rate=relaxation_rate_from_primitive(prim,rheology,p)
    chi=np.zeros_like(c)
    wet=c>0
    chi[wet]=rate[wet]*float(dx)/c[wet]
    return chi,rate,c

def blend_weight(chi):
    chi=np.asarray(chi,float)
    w=np.zeros_like(chi)
    mask=chi>2.0
    w[mask]=1.0-(2.0/chi[mask])**2
    return np.clip(w,0.0,1.0)

def filtered_cell_field(h):
    h=np.asarray(h,float)
    return -0.25*(np.roll(h,-1)-2*h+np.roll(h,1))

def blended_mass_flux(data,dx,rheology,p,diffusion_gate=None):
    qL=data['qL_face']; qR=data['qR_face']
    fL=hp.conservative_flux(qL,p,u_adv=data['uL_face'],primitive_values=data['primL_face'])
    fR=hp.conservative_flux(qR,p,u_adv=data['uR_face'],primitive_values=data['primR_face'])

    chiL,rL,cL=chi_from_primitive(data['primL_face'],dx,rheology,p)
    chiR,rR,cR=chi_from_primitive(data['primR_face'],dx,rheology,p)
    uL=np.asarray(data['uL_face']); uR=np.asarray(data['uR_face'])

    # Original HP-PCCU wave speeds, retained for central advective averaging/path orientation.
    am=np.minimum(np.minimum(uL-cL,uR-cR),0.0)
    ap=np.maximum(np.maximum(uL+cL,uR+cR),0.0)
    den=ap-am
    safe=np.where(den>p.speed_eps,den,1.0)

    # Advective central-upwind part without jump diffusion.
    H=(ap[:,None]*fL-am[:,None]*fR)/safe[:,None]
    deg=den<=p.speed_eps
    if np.any(deg):
        H[deg]=0.5*(fL[deg]+fR[deg])

    # Original CU jump coefficient nu_HP = -ap*am/(ap-am) >=0.
    nu_hp=np.zeros_like(den)
    ok=den>p.speed_eps
    nu_hp[ok]=(-ap[ok]*am[ok])/den[ok]

    # Face blend weight from neighboring reconstructed states.
    chiF=np.maximum(chiL,chiR)
    w=blend_weight(chiF)

    # AP high-frequency coefficient: target strong-limit nu_AP ~ D/dx,
    # since the filtered jump contributes the additional sin^4 damping.
    rateF=np.maximum(rL,rR)
    cF=np.maximum(cL,cR)
    D=np.zeros_like(cF)
    mask=rateF>0
    D[mask]=cF[mask]**2/rateF[mask]
    nu_ap=D/float(dx)

    # Standard state jump diffusion for all conservative variables, faded out.
    jump=qR-qL
    gate=np.ones_like(w) if diffusion_gate is None else np.asarray(diffusion_gate,float)
    H += -(gate*(1.0-w)*nu_hp)[:,None]*jump

    # AP filtered correction only for transported mass-like components.
    # Build a face-local high-pass proxy from cell-centered h using nearest
    # cell data available in qL_cell/qR_cell later; here use face h jump
    # curvature proxy through neighboring reconstructed eta/h differences
    # supplied by caller via data if available.
    if 'h_filter_jump_face' in data:
        dfilt=np.asarray(data['h_filter_jump_face'])
        H[:,0] += -(gate*w*nu_ap)*dfilt
        if H.shape[1] > 2:
            # transport composition with same filtered carrier proportionally
            hface=0.5*(data['primL_face'][3]+data['primR_face'][3])
            q2face=0.5*(qL[:,2]+qR[:,2])
            frac=np.where(hface>p.dry_h_tol,q2face/hface,0.0)
            H[:,2] += -(gate*w*nu_ap)*dfilt*frac

    if p.zero_transport_at_rest:
        huL,huR=data['huL_face'],data['huR_face']
        resting=((huL==0)&(huR==0)) if p.rest_flux_tol==0 else ((np.abs(huL)<=p.rest_flux_tol)&(np.abs(huR)<=p.rest_flux_tol))
        # Preserve the original HP-PCCU exact-rest transport rule.
        H[resting,0]=0.0
        if H.shape[1]>2:
            H[resting,2]=0.0

    return H,am,ap,w,nu_hp,nu_ap,cL,cR,rL,rR

def prepare_cache(q,B,dx,t,dt,p,boundary,rheology,q_is_sanitized=False,slope_geometry=None,diffusion_gate=None):
    q=np.asarray(q,float) if q_is_sanitized else hp.sanitize(q,p)
    if slope_geometry is None and (p.slope_correction or p.curvature_term):
        slope_geometry=hp.build_slope_geometry_1d(B,dx)
    base=hp.prepare_stage_face_cache(q,B,t,p,boundary,q_is_sanitized=True,slope_geometry=slope_geometry)
    data=base['data']

    # Build cell-centered high-pass field and its face jump.
    primq=data['prim_q']
    hc=np.asarray(primq[3],float)
    filt=filtered_cell_field(hc)
    # transmissive extension for filter field
    fext=np.concatenate(([filt[0]],filt,[filt[-1]]))
    # face between left/right neighboring cells
    dfilt=fext[1:]-fext[:-1]
    # number of physical faces = n+1
    data['h_filter_jump_face']=dfilt

    H,am,ap,w,nu_hp,nu_ap,cL,cR,rL,rR=blended_mass_flux(data,dx,rheology,p,diffusion_gate=diffusion_gate)

    Bpsi=hp.path_hydrostatic_source(data['qL_face'],data['qR_face'],data['etaL_face'],data['etaR_face'],p,
        primitive_L=data['primL_face'],primitive_R=data['primR_face'],grav_coeff=base['G_face'])
    Bj=hp.path_hydrostatic_source(data['qL_cell'],data['qR_cell'],data['etaL_cell'],data['etaR_cell'],p,
        grav_coeff=base['G_face'][:-1],grav_coeff_right=base['G_face'][1:])
    zero=np.zeros_like(Bpsi)
    GLh,GRh=hp.pccu_effective_face_values(zero,Bpsi,am,ap,p.speed_eps)
    return dict(q=q,data=data,Hadv=H,a_minus=am,a_plus=ap,Bpsi=Bpsi,Bj=Bj,GLh=GLh,GRh=GRh,
                w=w,nu_hp=nu_hp,nu_ap=nu_ap,cL=cL,cR=cR,rateL=rL,rateR=rR)

def rhs_adv(cache,dx):
    return -(cache['Hadv'][1:]-cache['Hadv'][:-1])/float(dx)

def rhs_hydro(cache,dx):
    R=-(cache['GLh'][1:]-cache['GRh'][:-1])/float(dx)+cache['Bj']/float(dx)
    R[:,0]=0.; R[:,2]=0.
    return R

def relax_newtonian(q_forced,dt,p,rheology):
    q=hp.sanitize(q_forced,p)
    prim=hp.primitive(q,p)
    h=np.asarray(prim[3]); rho=np.asarray(prim[4])
    rate=np.zeros_like(h)
    wet=h>p.dry_h_tol
    if rheology.enabled and rheology.mu>0:
        rate[wet]=3.0*rheology.mu/(rho[wet]*h[wet]**2)
    q[:,1]=np.where(wet,q[:,1]/(1.0+dt*rate),0.0)
    return hp.sanitize(q,p),rate

def cfl_dt(q,B,dx,t,dt_cap,p,boundary,rheology):
    cache=prepare_cache(q,B,dx,t,0.0,p,boundary,rheology,q_is_sanitized=True)
    # stability speed from max of retained baseline advective/acoustic and AP coefficient,
    # blended consistently; use conservative upper bound
    # Effective stability speed. For symmetric states 2*nu_hp=c, so at w=0
    # this is exactly the original acoustic scale. In the AP regime the
    # low-frequency CU viscosity is replaced by the filtered AP viscosity.
    uface=np.maximum(np.abs(cache['data']['uL_face']),np.abs(cache['data']['uR_face']))
    speed_hp=np.maximum(np.abs(cache['a_minus']),np.abs(cache['a_plus']))
    speed_ap=uface + 2.0*cache['nu_ap']
    speed_face=(1.0-cache['w'])*speed_hp + cache['w']*speed_ap
    speed=float(np.max(speed_face))
    if speed<=1e-14:
        return float(dt_cap),cache,speed
    return min(float(dt_cap),0.99*p.cfl*float(dx)/speed),cache,speed

def step_imex_euler(q,B,dx,t,dt_cap,p,boundary,rheology=NewtonianRheology(),return_diagnostics=False,max_retries=20):
    qn=hp.sanitize(q,p)
    dt,cache_n,speed=cfl_dt(qn,B,dx,t,dt_cap,p,boundary,rheology)
    retries=0
    while True:
        cache_n=prepare_cache(qn,B,dx,t,dt,p,boundary,rheology,q_is_sanitized=True)
        Radv=rhs_adv(cache_n,dx)
        qstar_raw=qn+dt*Radv
        hraw=hp.raw_thickness(qstar_raw,p)
        tol=5e-13*max(1.0,float(np.max(cache_n['data']['prim_q'][3])))
        if float(np.min(hraw))>=-tol:
            qstar=hp.sanitize(qstar_raw,p)
            cache_s=prepare_cache(qstar,B,dx,t+dt,dt,p,boundary,rheology,q_is_sanitized=True)
            Rh=rhs_hydro(cache_s,dx)
            qforced=qstar.copy(); qforced[:,1]+=dt*Rh[:,1]
            qnew,rate=relax_newtonian(qforced,dt,p,rheology)
            qnew[:,0]=qstar[:,0]; qnew[:,2]=qstar[:,2]
            qnew=hp.sanitize(qnew,p)
            diag=dict(dt=dt,retries=retries,speed_max=speed,
                      w_max=float(np.max(cache_n['w'])),w_mean=float(np.mean(cache_n['w'])),
                      mass_change_hydro_rheology=float(np.max(np.abs(qnew[:,0]-qstar[:,0]))))
            return (qnew,diag) if return_diagnostics else qnew
        retries+=1
        if retries>max_retries:
            raise RuntimeError("positivity retry failed")
        dt*=0.5
