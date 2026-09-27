"""IMEX/AP layer for the pure-liquid 2-D HP-PCCU reference core.

This module adds Newtonian local relaxation, relaxation-aware blended CU
dissipation, split inertial/hydrostatic PCCU residuals, and local implicit
momentum relaxation.
"""
from dataclasses import dataclass
import numpy as np
import hp_pccu_2d_core as hp


@dataclass(frozen=True)
class NewtonianRheology:
    mu: float = 1.0e5
    enabled: bool = True


def relaxation_rate_from_h(h, rheology, p=hp.Params()):
    h=np.asarray(h,float)
    out=np.zeros_like(h)
    wet=h>p.dry_h_tol
    if rheology.enabled and rheology.mu>0.0:
        out[wet]=3.0*rheology.mu/(p.rho*h[wet]**2)
    return out


def blend_weight(chi):
    chi=np.asarray(chi,float)
    w=np.zeros_like(chi)
    m=chi>2.0
    w[m]=1.0-(2.0/chi[m])**2
    return np.clip(w,0.0,1.0)


def _filtered_cell_field(h, axis):
    """Directional high-pass field F=-1/4 second difference, transmissive edges."""
    h=np.asarray(h,float)
    F=np.zeros_like(h)
    if axis==1:  # x
        if h.shape[1]>=3:
            F[:,1:-1]=-0.25*(h[:,2:]-2.0*h[:,1:-1]+h[:,:-2])
            F[:,0]=F[:,1]
            F[:,-1]=F[:,-2]
    else:        # y
        if h.shape[0]>=3:
            F[1:-1,:]=-0.25*(h[2:,:]-2.0*h[1:-1,:]+h[:-2,:])
            F[0,:]=F[1,:]
            F[-1,:]=F[-2,:]
    return F


def _filtered_face_jump(h, direction):
    if direction=='x':
        F=_filtered_cell_field(h,1)
        ny,nx=h.shape
        dF=np.zeros((ny,nx+1))
        dF[:,1:nx]=F[:,1:]-F[:,:-1]
        return dF
    F=_filtered_cell_field(h,0)
    ny,nx=h.shape
    dF=np.zeros((ny+1,nx))
    dF[1:ny,:]=F[1:,:]-F[:-1,:]
    return dF


def blended_cu_flux(qL,qR,direction,dcell,dF,rheology,p=hp.Params()):
    """Original HP-PCCU for chi<=2; AP-filtered replacement for chi>2."""
    fL=hp.inertial_flux(qL,direction,p)
    fR=hp.inertial_flux(qR,direction,p)
    am,ap=hp.face_speeds(qL,qR,direction,p)
    den=ap-am
    H=np.zeros_like(fL)
    ok=den>p.speed_eps
    H[ok]=(ap[ok,None]*fL[ok]-am[ok,None]*fR[ok])/den[ok,None]
    H[~ok]=0.5*(fL[~ok]+fR[~ok])

    nu_hp=np.zeros_like(den)
    nu_hp[ok]=(-ap[ok]*am[ok])/den[ok]

    hL,_,_=hp.primitive(qL,p); hR,_,_=hp.primitive(qR,p)
    gL=relaxation_rate_from_h(hL,rheology,p)
    gR=relaxation_rate_from_h(hR,rheology,p)
    cL=np.sqrt(np.maximum(p.g*hL,0.0)); cR=np.sqrt(np.maximum(p.g*hR,0.0))
    chiL=np.zeros_like(hL);chiR=np.zeros_like(hR)
    m=cL>0;chiL[m]=gL[m]*dcell/cL[m]
    m=cR>0;chiR[m]=gR[m]*dcell/cR[m]
    w=blend_weight(np.maximum(chiL,chiR))
    # The AP diffusion-limit derivation assumes a smooth wet state.  Keep the
    # original HP-PCCU stabilization unchanged across wet/dry and strongly
    # under-resolved thickness jumps, where it is needed for positivity.
    hmax=np.maximum(hL,hR)
    hmin=np.minimum(hL,hR)
    wet_ratio=np.zeros_like(hmax)
    m=hmax>p.dry_h_tol
    wet_ratio[m]=hmin[m]/hmax[m]
    w=np.where(wet_ratio>=0.10,w,0.0)

    jump=qR-qL
    H -= ((1.0-w)*nu_hp)[...,None]*jump

    rateF=np.maximum(gL,gR); cF=np.maximum(cL,cR)
    D=np.zeros_like(rateF)
    m=rateF>0
    D[m]=cF[m]**2/rateF[m]
    nu_ap=D/dcell
    H[...,0] -= w*nu_ap*dF

    if p.zero_transport_at_rest:
        _,uL,vL=hp.primitive(qL,p); _,uR,vR=hp.primitive(qR,p)
        unL=uL if direction=='x' else vL
        unR=uR if direction=='x' else vR
        resting=(hL*unL==0.0)&(hR*unR==0.0) if p.rest_flux_tol==0.0 else \
                ((np.abs(hL*unL)<=p.rest_flux_tol)&(np.abs(hR*unR)<=p.rest_flux_tol))
        # Exact baseline rest rule.  For an equilibrium smooth h field dF may be nonzero;
        # suppressing mass transport here is required by the Fortran baseline behavior.
        H[resting,0]=0.0

    speed_hp=np.maximum(np.abs(am),np.abs(ap))
    hface_u=np.maximum(np.abs((hp.primitive(qL,p)[1] if direction=='x' else hp.primitive(qL,p)[2])),
                       np.abs((hp.primitive(qR,p)[1] if direction=='x' else hp.primitive(qR,p)[2])))
    speed_ap=hface_u+2.0*nu_ap
    speed_face=(1.0-w)*speed_hp+w*speed_ap
    return H,am,ap,w,nu_hp,nu_ap,speed_face


def split_residuals(q,B_vertex,dx,dy,rheology=NewtonianRheology(),p=hp.Params(),return_diagnostics=False):
    """Return explicit inertial residual and hydrostatic PCCU residual separately."""
    q=hp.sanitize(q,p)
    rec=hp.reconstruct(q,B_vertex,p)
    ny,nx=q.shape[:2]
    hc=hp.primitive(q,p)[0]

    # x faces
    dFx=_filtered_face_jump(hc,'x')
    Hx,amx,apx,wx,nuhpx,nuapx,speedx=blended_cu_flux(
        rec['xL'],rec['xR'],'x',dx,dFx,rheology,p)
    hL,_,_=hp.primitive(rec['xL'],p); hR,_,_=hp.primitive(rec['xR'],p)
    Px=hp.hydrostatic_path(hL,rec['eta_xL'],hR,rec['eta_xR'],'x',p)
    denx=apx-amx
    HLx=np.zeros_like(Px); HRx=np.zeros_like(Px); ok=denx>p.speed_eps
    HLx[ok]=(amx[ok]/denx[ok])[...,None]*Px[ok]
    HRx[ok]=(apx[ok]/denx[ok])[...,None]*Px[ok]

    # y faces
    dFy=_filtered_face_jump(hc,'y')
    Hy,amy,apy,wy,nuhpy,nuapy,speedy=blended_cu_flux(
        rec['yB'],rec['yT'],'y',dy,dFy,rheology,p)
    hB,_,_=hp.primitive(rec['yB'],p); hT,_,_=hp.primitive(rec['yT'],p)
    Py=hp.hydrostatic_path(hB,rec['eta_yB'],hT,rec['eta_yT'],'y',p)
    deny=apy-amy
    HLy=np.zeros_like(Py); HRy=np.zeros_like(Py); ok=deny>p.speed_eps
    HLy[ok]=(amy[ok]/deny[ok])[...,None]*Py[ok]
    HRy[ok]=(apy[ok]/deny[ok])[...,None]*Py[ok]

    # cell paths
    hW,_,_=hp.primitive(rec['qW'],p); hE,_,_=hp.primitive(rec['qE'],p)
    Pcx=hp.hydrostatic_path(hW,rec['etaW'],hE,rec['etaE'],'x',p)
    hS,_,_=hp.primitive(rec['qS'],p); hN,_,_=hp.primitive(rec['qN'],p)
    Pcy=hp.hydrostatic_path(hS,rec['etaS'],hN,rec['etaN'],'y',p)

    div_adv=(Hx[:,1:]-Hx[:,:-1])/dx + (Hy[1:]-Hy[:-1])/dy
    div_hyd=(HLx[:,1:]-HRx[:,:-1]-Pcx)/dx + (HLy[1:]-HRy[:-1]-Pcy)/dy
    Radv=-div_adv
    Rhyd=-div_hyd
    # Hydrostatic path must never modify mass.
    Rhyd[...,0]=0.0

    # Preserve an exact global lake-at-rest state under operator splitting.
    # The unsplit Fortran-rebased residual cancels this equilibrium to roundoff;
    # splitting can otherwise accumulate that roundoff over many IMEX steps.
    h0,u0,v0=hp.primitive(q,p)
    eta0=h0+hp.geometry_from_vertices(B_vertex)['B_cent']
    escale=max(1.0,float(np.max(np.abs(eta0))))
    etol=512.0*np.finfo(float).eps*escale
    if float(np.max(np.abs(u0)))==0.0 and float(np.max(np.abs(v0)))==0.0 and \
       float(np.max(eta0)-np.min(eta0))<=etol:
        Radv.fill(0.0); Rhyd.fill(0.0)

    if return_diagnostics:
        return Radv,Rhyd,dict(reconstruction=rec,wx=wx,wy=wy,
                              max_w=float(max(np.max(wx),np.max(wy))),
                              mean_w=float(0.5*(np.mean(wx)+np.mean(wy))),
                              max_speed_x=float(np.max(speedx)),
                              max_speed_y=float(np.max(speedy)))
    return Radv,Rhyd


def relax_newtonian(q_forced,dt,rheology,p=hp.Params()):
    q=hp.sanitize(q_forced,p)
    h,_,_=hp.primitive(q,p)
    gamma=relaxation_rate_from_h(h,rheology,p)
    fac=1.0/(1.0+dt*gamma)
    q[...,1]*=fac
    q[...,2]*=fac
    return hp.sanitize(q,p),gamma


def max_dt(q,B_vertex,dx,dy,rheology=NewtonianRheology(),p=hp.Params()):
    _,_,d=split_residuals(q,B_vertex,dx,dy,rheology,p,True)
    denom=(d['max_speed_x']/dx if d['max_speed_x']>0 else 0.0)+(d['max_speed_y']/dy if d['max_speed_y']>0 else 0.0)
    return np.inf if denom<=0 else p.cfl/denom


def step_imex_euler(q,B_vertex,dx,dy,t,dt_cap,rheology=NewtonianRheology(),p=hp.Params(),max_retries=25):
    """Explicit inertia -> known HP hydro forcing -> local BE rheology."""
    qn=hp.sanitize(q,p)
    dt=min(float(dt_cap),0.99*max_dt(qn,B_vertex,dx,dy,rheology,p))
    retries=0
    tol=5e-13*max(1.0,float(np.max(hp.primitive(qn,p)[0])))
    while True:
        Radv,_,d0=split_residuals(qn,B_vertex,dx,dy,rheology,p,True)
        qstar_raw=qn+dt*Radv
        if np.min(qstar_raw[...,0]/p.rho)>=-tol:
            qstar=hp.sanitize(qstar_raw,p)
            # Hydrostatic forcing evaluated using the updated explicit mass/state.
            _,Rhyd,dh=split_residuals(qstar,B_vertex,dx,dy,rheology,p,True)
            qforced=qstar+dt*Rhyd
            qnew,gamma=relax_newtonian(qforced,dt,rheology,p)
            # Enforce exact operator split: hydro+rheology cannot alter mass.
            qnew[...,0]=qstar[...,0]
            qnew=hp.sanitize(qnew,p)
            return qnew,dict(dt=dt,retries=retries,max_w=max(d0['max_w'],dh['max_w']),
                             mean_w=d0['mean_w'],max_gamma_dt=float(np.max(gamma*dt)),
                             max_speed_x=d0['max_speed_x'],max_speed_y=d0['max_speed_y'])
        retries+=1
        if retries>max_retries:
            raise RuntimeError('IMEX Euler positivity retry limit exceeded')
        dt*=0.5


def run(q0,B_vertex,dx,dy,t_end,rheology=NewtonianRheology(),p=hp.Params(),output_times=(),dt_cap=np.inf):
    q=hp.sanitize(q0,p);t=0.0
    outs=sorted(set([float(z) for z in output_times if 0<=z<=t_end]+[float(t_end)]));snaps=[];k=0
    if outs and outs[0]<=1e-14:snaps.append((outs[0],q.copy()));k=1
    stats={'steps':0,'retries':0,'min_dt':np.inf,'max_dt':0.0,'sum_dt':0.0,'max_w':0.0}
    while t<t_end-1e-14:
        cap=min(float(dt_cap),t_end-t)
        if k<len(outs):cap=min(cap,max(outs[k]-t,0.0))
        if cap<=1e-15:
            snaps.append((outs[k],q.copy()));k+=1;continue
        q,sd=step_imex_euler(q,B_vertex,dx,dy,t,cap,rheology,p);t+=sd['dt']
        stats['steps']+=1;stats['retries']+=sd['retries'];stats['sum_dt']+=sd['dt']
        stats['min_dt']=min(stats['min_dt'],sd['dt']);stats['max_dt']=max(stats['max_dt'],sd['dt']);stats['max_w']=max(stats['max_w'],sd['max_w'])
        while k<len(outs) and t>=outs[k]-1e-12:
            snaps.append((outs[k],q.copy()));k+=1
    stats['mean_dt']=stats['sum_dt']/max(stats['steps'],1)
    return snaps,stats
