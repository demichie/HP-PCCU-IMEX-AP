"""Pure-liquid 2-D HP-PCCU reference core used by the manuscript benchmarks.

The implementation contains the hydrostatic-path PCCU spatial method on a
Cartesian grid with continuous topography represented by shared Q1 vertex
geometry and in-cell face traces.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Params:
    rho: float = 1000.0
    g: float = 9.81
    cfl: float = 0.40
    limiter: int = 3
    theta: float = 1.3
    reconstr_coeff: float = 1.0
    dry_h_tol: float = 1.0e-10
    speed_eps: float = 1.0e-14
    hp_bed_step_tol: float = 1.0e-12
    hp_dynamic_residual_threshold: float = 0.90
    zero_transport_at_rest: bool = True
    rest_flux_tol: float = 0.0


# -----------------------------------------------------------------------------
# Basic state conversion / admissibility
# -----------------------------------------------------------------------------
def sanitize(q, p=Params()):
    q = np.asarray(q, dtype=float).copy()
    q[..., 0] = np.maximum(q[..., 0], 0.0)
    h = q[..., 0] / p.rho
    dry = h <= p.dry_h_tol
    q[..., 0] = np.where(dry, 0.0, q[..., 0])
    q[..., 1] = np.where(dry, 0.0, q[..., 1])
    q[..., 2] = np.where(dry, 0.0, q[..., 2])
    return q


def primitive(q, p=Params()):
    q = np.asarray(q, dtype=float)
    M = np.maximum(q[..., 0], 0.0)
    h = M / p.rho
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    wet = h > p.dry_h_tol
    u[wet] = q[..., 1][wet] / M[wet]
    v[wet] = q[..., 2][wet] / M[wet]
    return h, u, v


def state_from_huv(h, u, v, p=Params()):
    h, u, v = np.broadcast_arrays(np.asarray(h,float), np.asarray(u,float), np.asarray(v,float))
    h = np.maximum(h, 0.0)
    wet = h > p.dry_h_tol
    q = np.zeros(h.shape + (3,), dtype=float)
    q[...,0] = p.rho*h
    q[...,1] = np.where(wet, p.rho*h*u, 0.0)
    q[...,2] = np.where(wet, p.rho*h*v, 0.0)
    return q


def safe_velocity(hu, h, p=Params()):
    hu, h = np.broadcast_arrays(np.asarray(hu,float), np.asarray(h,float))
    u = np.zeros_like(h)
    wet = h > p.dry_h_tol
    u[wet] = hu[wet]/h[wet]
    return u


# -----------------------------------------------------------------------------
# Limiter and limited reconstruction
# -----------------------------------------------------------------------------
def minmod(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return np.where(a*b > 0.0, np.sign(a)*np.minimum(np.abs(a),np.abs(b)), 0.0)


def limited_slope_1d(v, p=Params(), boundary='zero'):
    """Fortran geometry_2d limit() on unit-spaced data.

    HP eta uses zero boundary slope. For ordinary cell-centred
    reconstructions we also use zero slope at the external transmissive boundary;
    this does not affect the interior benchmarks for which the core is intended.
    """
    v = np.asarray(v,float)
    s = np.zeros_like(v)
    if v.size >= 3 and p.limiter != 0:
        dm = v[1:-1]-v[:-2]
        dp = v[2:]-v[1:-1]
        dc = 0.5*(v[2:]-v[:-2])
        if p.limiter == 3:
            s[1:-1] = minmod(dc, p.theta*minmod(dm,dp))
        elif p.limiter == 5:
            s[1:-1] = dc
        else:
            s[1:-1] = minmod(dm,dp)
    return p.reconstr_coeff*s


def reconstruct_direct_1d(v, p=Params()):
    s = limited_slope_1d(v,p)
    return np.asarray(v,float)-0.5*s, np.asarray(v,float)+0.5*s


def geometry_from_vertices(B_vertex):
    """Fortran geometry_2d: B_vertex is authoritative; derive Q1 centers/faces."""
    V=np.asarray(B_vertex,float)
    if V.ndim!=2 or min(V.shape)<2:
        raise ValueError("B_vertex must have shape (ny+1,nx+1)")
    B_face_x=0.5*(V[:-1,:]+V[1:,:])       # (ny,nx+1), vertical faces
    B_face_y=0.5*(V[:,:-1]+V[:,1:])       # (ny+1,nx), horizontal faces
    B_cent=0.25*(V[:-1,:-1]+V[:-1,1:]+V[1:,:-1]+V[1:,1:])
    return dict(B_vertex=V,B_cent=B_cent,B_face_x=B_face_x,B_face_y=B_face_y)



# -----------------------------------------------------------------------------
# Latest dry-safe HP reconstruction (direct translation of reconstruct_hp_line)
# -----------------------------------------------------------------------------
def reconstruct_hp_line(h_center,u_center,B_minus,B_plus,
                        h_minus_direct,h_plus_direct,
                        hu_minus_direct,hu_plus_direct,
                        u_minus_candidate,u_plus_candidate,
                        hydrostatic_residual,topographic_relief_ratio,
                        p=Params()):
    h_center=np.asarray(h_center,float); u_center=np.asarray(u_center,float)
    B_minus=np.asarray(B_minus,float); B_plus=np.asarray(B_plus,float)
    n=h_center.size
    eta_center=h_center+0.5*(B_minus+B_plus)
    u_safe=u_center.copy(); u_safe[h_center<=p.dry_h_tol]=0.0

    eta_minus=np.empty(n); eta_plus=np.empty(n)
    hm_eta=np.empty(n); hp_eta=np.empty(n)
    eta_slope=limited_slope_1d(eta_center,p)
    # Eta boundary convention.
    if n:
        eta_slope[0]=0.0; eta_slope[-1]=0.0
    slope_min=2.0*(B_plus-eta_center)
    slope_max=2.0*(eta_center-B_minus)
    eta_slope=np.minimum(np.maximum(eta_slope,slope_min),slope_max)
    eta_minus=eta_center-0.5*eta_slope
    eta_plus=eta_center+0.5*eta_slope
    hm_eta=np.maximum(eta_minus-B_minus,0.0)
    hp_eta=np.maximum(eta_plus-B_plus,0.0)

    hm_orig=np.maximum(np.asarray(h_minus_direct,float),0.0)
    hp_orig=np.maximum(np.asarray(h_plus_direct,float),0.0)
    Eh=np.zeros(n); Eeta=np.zeros(n)
    if n>1:
        dh=np.abs(hp_orig[:-1]-hm_orig[1:])
        de=np.abs(hp_eta[:-1]-hm_eta[1:])
        Eh[:-1]+=dh; Eh[1:]+=dh
        Eeta[:-1]+=de; Eeta[1:]+=de

    w=np.empty(n)
    for i in range(n):
        den=Eh[i]+Eeta[i]
        tol=1e-14*max(1.0,h_center[i])
        w[i]=Eh[i]/den if den>tol else 0.5
        if h_center[i] <= p.dry_h_tol:
            w[i]=1.0
        elif hm_eta[i] <= p.dry_h_tol or hp_eta[i] <= p.dry_h_tol:
            w[i]=0.0 if hydrostatic_residual[i] > p.hp_dynamic_residual_threshold else 1.0
        elif (hydrostatic_residual[i] > p.hp_dynamic_residual_threshold and
              topographic_relief_ratio[i] > 1.0):
            w[i]=0.0

    hm=(1.0-w)*hm_orig+w*hm_eta
    hp=(1.0-w)*hp_orig+w*hp_eta

    hum=np.empty(n); hup=np.empty(n)
    for i in range(n):
        momentum_weight=w[i] if abs(B_plus[i]-B_minus[i])>p.hp_bed_step_tol else 0.0
        hum0=(1.0-momentum_weight)*hu_minus_direct[i] + momentum_weight*hm_eta[i]*u_minus_candidate[i]
        hup0=(1.0-momentum_weight)*hu_plus_direct[i] + momentum_weight*hp_eta[i]*u_plus_candidate[i]
        lo=max(0,i-1); hi=min(n,i+2)
        umin=np.min(u_safe[lo:hi]); umax=np.max(u_safe[lo:hi])
        dm_target=0.5*(hup0-hum0)-u_safe[i]*0.5*(hp[i]-hm[i])
        dm_lower=max((u_safe[i]-umax)*hm[i], (umin-u_safe[i])*hp[i])
        dm_upper=min((u_safe[i]-umin)*hm[i], (umax-u_safe[i])*hp[i])
        dm=min(max(dm_target,dm_lower),dm_upper)
        hum[i]=u_safe[i]*hm[i]-dm
        hup[i]=u_safe[i]*hp[i]+dm
    return hm,hp,hum,hup,eta_minus,eta_plus,w


def hydrostatic_residual_2d(h,B):
    """Exact four-neighbour dimensionless residual from reconstruction_2d.f90."""
    h=np.asarray(h,float); B=np.asarray(B,float); eta=h+B
    ny,nx=h.shape; out=np.zeros_like(h)
    eps=np.finfo(float).eps
    for j in range(ny):
        for i in range(nx):
            num=0.0; den=0.0
            for jj,ii in ((j,i-1),(j,i+1),(j-1,i),(j+1,i)):
                if 0<=jj<ny and 0<=ii<nx:
                    num += abs(eta[jj,ii]-eta[j,i])
                    den += abs(h[jj,ii]-h[j,i])+abs(B[jj,ii]-B[j,i])
            scale=128.0*eps*max(1.0,abs(eta[j,i]),abs(h[j,i]),abs(B[j,i]))
            out[j,i]=min(1.0,max(0.0,num/den)) if den>scale else 0.0
    return out


# -----------------------------------------------------------------------------
# Complete 2-D reconstruction workspace for pure liquid
# -----------------------------------------------------------------------------
def reconstruct(q,B_vertex,p=Params()):
    q=sanitize(q,p); geom=geometry_from_vertices(B_vertex); B=geom['B_cent']
    h,u,v=primitive(q,p); ny,nx=h.shape
    if B.shape!=(ny,nx): raise ValueError("B_vertex dimensions incompatible with q")
    Bfx=geom['B_face_x']; Bfy=geom['B_face_y']
    BW=Bfx[:,:-1]; BE=Bfx[:,1:]; BS=Bfy[:-1,:]; BN=Bfy[1:,:]
    residual=hydrostatic_residual_2d(h,B)
    relief=np.maximum.reduce([BW,BE,BS,BN])-np.minimum.reduce([BW,BE,BS,BN])
    relief_ratio=relief/np.maximum(h,p.dry_h_tol)

    # Direct candidates.
    hW=np.empty_like(h);hE=np.empty_like(h);huW=np.empty_like(h);huE=np.empty_like(h)
    uW=np.empty_like(h);uE=np.empty_like(h);vW=np.empty_like(h);vE=np.empty_like(h)
    hS=np.empty_like(h);hN=np.empty_like(h);hvS=np.empty_like(h);hvN=np.empty_like(h)
    vS=np.empty_like(h);vN=np.empty_like(h);uS=np.empty_like(h);uN=np.empty_like(h)
    hu=h*u; hv=h*v
    for j in range(ny):
        hW[j],hE[j]=reconstruct_direct_1d(h[j],p)
        huW[j],huE[j]=reconstruct_direct_1d(hu[j],p)
        uW[j],uE[j]=reconstruct_direct_1d(u[j],p)
        vW[j],vE[j]=reconstruct_direct_1d(v[j],p)
    for i in range(nx):
        hS[:,i],hN[:,i]=reconstruct_direct_1d(h[:,i],p)
        hvS[:,i],hvN[:,i]=reconstruct_direct_1d(hv[:,i],p)
        vS[:,i],vN[:,i]=reconstruct_direct_1d(v[:,i],p)
        uS[:,i],uN[:,i]=reconstruct_direct_1d(u[:,i],p)

    etaW=np.empty_like(h);etaE=np.empty_like(h);wx=np.empty_like(h)
    etaS=np.empty_like(h);etaN=np.empty_like(h);wy=np.empty_like(h)
    for j in range(ny):
        vals=reconstruct_hp_line(h[j],u[j],BW[j],BE[j],hW[j],hE[j],huW[j],huE[j],uW[j],uE[j],residual[j],relief_ratio[j],p)
        hW[j],hE[j],huW[j],huE[j],etaW[j],etaE[j],wx[j]=vals
        uW[j]=safe_velocity(huW[j],hW[j],p);uE[j]=safe_velocity(huE[j],hE[j],p)
    for i in range(nx):
        vals=reconstruct_hp_line(h[:,i],v[:,i],BS[:,i],BN[:,i],hS[:,i],hN[:,i],hvS[:,i],hvN[:,i],vS[:,i],vN[:,i],residual[:,i],relief_ratio[:,i],p)
        hS[:,i],hN[:,i],hvS[:,i],hvN[:,i],etaS[:,i],etaN[:,i],wy[:,i]=vals
        vS[:,i]=safe_velocity(hvS[:,i],hS[:,i],p);vN[:,i]=safe_velocity(hvN[:,i],hN[:,i],p)

    # Final transverse volumetric momenta follow Fortran: h_final * transverse candidate velocity.
    qW=state_from_huv(hW,uW,vW,p); qE=state_from_huv(hE,uE,vE,p)
    qS=state_from_huv(hS,uS,vS,p); qN=state_from_huv(hN,uN,vN,p)

    # Oriented interface arrays. x faces shape (ny,nx+1), y faces (ny+1,nx).
    xL=np.empty((ny,nx+1,3)); xR=np.empty_like(xL)
    exL=np.empty((ny,nx+1)); exR=np.empty_like(exL)
    # left/right states at a face = E of left cell / W of right cell.
    xL[:,1:nx]=qE[:,:-1]; xR[:,1:nx]=qW[:,1:]
    exL[:,1:nx]=etaE[:,:-1]; exR[:,1:nx]=etaW[:,1:]
    # transmissive external ghost convention
    xL[:,0]=qW[:,0];xR[:,0]=qW[:,0];exL[:,0]=etaW[:,0];exR[:,0]=etaW[:,0]
    xL[:,nx]=qE[:,-1];xR[:,nx]=qE[:,-1];exL[:,nx]=etaE[:,-1];exR[:,nx]=etaE[:,-1]

    yB=np.empty((ny+1,nx,3)); yT=np.empty_like(yB)
    eyB=np.empty((ny+1,nx)); eyT=np.empty_like(eyB)
    yB[1:ny]=qN[:-1]; yT[1:ny]=qS[1:]
    eyB[1:ny]=etaN[:-1]; eyT[1:ny]=etaS[1:]
    yB[0]=qS[0];yT[0]=qS[0];eyB[0]=etaS[0];eyT[0]=etaS[0]
    yB[ny]=qN[-1];yT[ny]=qN[-1];eyB[ny]=etaN[-1];eyT[ny]=etaN[-1]

    return dict(qW=qW,qE=qE,qS=qS,qN=qN,etaW=etaW,etaE=etaE,etaS=etaS,etaN=etaN,
                xL=xL,xR=xR,eta_xL=exL,eta_xR=exR,
                yB=yB,yT=yT,eta_yB=eyB,eta_yT=eyT,
                BW=BW,BE=BE,BS=BS,BN=BN,w_eta_x=wx,w_eta_y=wy,
                hydrostatic_residual=residual,topographic_relief_ratio=relief_ratio)


# -----------------------------------------------------------------------------
# PCCU algebra
# -----------------------------------------------------------------------------
def inertial_flux(q,direction,p=Params()):
    h,u,v=primitive(q,p)
    un=u if direction=='x' else v
    return un[...,None]*q


def face_speeds(qL,qR,direction,p=Params()):
    hL,uL,vL=primitive(qL,p); hR,uR,vR=primitive(qR,p)
    nL=uL if direction=='x' else vL; nR=uR if direction=='x' else vR
    cL=np.sqrt(np.maximum(p.g*hL,0.0)); cR=np.sqrt(np.maximum(p.g*hR,0.0))
    am=np.minimum(np.minimum(nL-cL,nR-cR),0.0)
    ap=np.maximum(np.maximum(nL+cL,nR+cR),0.0)
    return am,ap


def central_upwind_flux(qL,qR,direction,p=Params()):
    fL=inertial_flux(qL,direction,p); fR=inertial_flux(qR,direction,p)
    am,ap=face_speeds(qL,qR,direction,p); den=ap-am
    H=np.zeros_like(fL); ok=den>p.speed_eps
    H[ok]=(ap[ok,None]*fL[ok]-am[ok,None]*fR[ok]+(ap[ok]*am[ok])[:,None]*(qR[ok]-qL[ok]))/den[ok,None]
    H[~ok]=0.5*(fL[~ok]+fR[~ok])
    if p.zero_transport_at_rest:
        hL,uL,vL=primitive(qL,p);hR,uR,vR=primitive(qR,p)
        unL=uL if direction=='x' else vL;unR=uR if direction=='x' else vR
        huL=hL*unL;huR=hR*unR
        resting=(huL==0.0)&(huR==0.0) if p.rest_flux_tol==0 else ((np.abs(huL)<=p.rest_flux_tol)&(np.abs(huR)<=p.rest_flux_tol))
        H[resting,0]=0.0
    return H,am,ap


def regularize_dry_gamma_pair(hL,gL,hR,gR,p=Params()):
    gL=np.asarray(gL,float).copy();gR=np.asarray(gR,float).copy()
    leftdry=(hL<=p.dry_h_tol)&(hR>p.dry_h_tol)
    rightdry=(hR<=p.dry_h_tol)&(hL>p.dry_h_tol)
    gL[leftdry]=gR[leftdry];gR[rightdry]=gL[rightdry]
    return gL,gR


def hydrostatic_path(hL,etaL,hR,etaR,direction,p=Params()):
    """Pure-liquid reduction of the Fortran 3-point path integral.

    Gamma=rho*g is constant, so the exact straight-path integral is
      -Gamma * 0.5*(hL+hR)*(etaR-etaL)
    in the normal momentum component.
    """
    gamma=p.rho*p.g
    val=-gamma*0.5*(hL+hR)*(etaR-etaL)
    P=np.zeros(hL.shape+(3,),float)
    P[...,1 if direction=='x' else 2]=val
    return P


def oriented_pair(H,P,am,ap,p=Params()):
    den=ap-am; GL=H.copy();GR=H.copy();ok=den>p.speed_eps
    GL[ok]=H[ok]+(am[ok]/den[ok])[:,None]*P[ok]
    GR[ok]=H[ok]+(ap[ok]/den[ok])[:,None]*P[ok]
    return GL,GR


def spatial_residual(q,B_vertex,dx,dy,p=Params(),return_diagnostics=False):
    q=sanitize(q,p); rec=reconstruct(q,B_vertex,p); ny,nx=q.shape[:2]

    # x interfaces
    Hx,amx,apx=central_upwind_flux(rec['xL'].reshape(-1,3),rec['xR'].reshape(-1,3),'x',p)
    Hx=Hx.reshape(ny,nx+1,3);amx=amx.reshape(ny,nx+1);apx=apx.reshape(ny,nx+1)
    hL,_,_=primitive(rec['xL'],p);hR,_,_=primitive(rec['xR'],p)
    Px=hydrostatic_path(hL,rec['eta_xL'],hR,rec['eta_xR'],'x',p)
    GLx,GRx=oriented_pair(Hx.reshape(-1,3),Px.reshape(-1,3),amx.ravel(),apx.ravel(),p)
    GLx=GLx.reshape(ny,nx+1,3);GRx=GRx.reshape(ny,nx+1,3)

    # y interfaces
    Hy,amy,apy=central_upwind_flux(rec['yB'].reshape(-1,3),rec['yT'].reshape(-1,3),'y',p)
    Hy=Hy.reshape(ny+1,nx,3);amy=amy.reshape(ny+1,nx);apy=apy.reshape(ny+1,nx)
    hB,_,_=primitive(rec['yB'],p);hT,_,_=primitive(rec['yT'],p)
    Py=hydrostatic_path(hB,rec['eta_yB'],hT,rec['eta_yT'],'y',p)
    GLy,GRy=oriented_pair(Hy.reshape(-1,3),Py.reshape(-1,3),amy.ravel(),apy.ravel(),p)
    GLy=GLy.reshape(ny+1,nx,3);GRy=GRy.reshape(ny+1,nx,3)

    # Cell hydrostatic paths W->E and S->N.
    hW,_,_=primitive(rec['qW'],p);hE,_,_=primitive(rec['qE'],p)
    Pcx=hydrostatic_path(hW,rec['etaW'],hE,rec['etaE'],'x',p)
    hS,_,_=primitive(rec['qS'],p);hN,_,_=primitive(rec['qN'],p)
    Pcy=hydrostatic_path(hS,rec['etaS'],hN,rec['etaN'],'y',p)

    # Fortran divFlux = (G_right_seen_by_left - G_left_seen_by_right - P_cell)/d.
    div=(GLx[:,1:]-GRx[:,:-1]-Pcx)/dx + (GLy[1:]-GRy[:-1]-Pcy)/dy
    R=-div
    if return_diagnostics:
        return R,dict(reconstruction=rec,Hx=Hx,Hy=Hy,Px=Px,Py=Py,
                      max_speed_x=float(np.max(np.maximum(np.abs(amx),np.abs(apx)))),
                      max_speed_y=float(np.max(np.maximum(np.abs(amy),np.abs(apy)))))
    return R


def max_dt(q,dx,dy,p=Params()):
    h,u,v=primitive(sanitize(q,p),p)
    c=np.sqrt(np.maximum(p.g*h,0.0))
    sx=float(np.max(np.abs(u)+c)); sy=float(np.max(np.abs(v)+c))
    denom=(sx/dx if sx>0 else 0.0)+(sy/dy if sy>0 else 0.0)
    return np.inf if denom<=0 else p.cfl/denom


def step_ssprk2(q,B_vertex,dx,dy,t,dt_cap,p=Params(),max_retries=20):
    q=sanitize(q,p); dt=min(float(dt_cap),0.99*max_dt(q,dx,dy,p)); retries=0
    tol=5e-13*max(1.0,float(np.max(primitive(q,p)[0])))
    while True:
        R0=spatial_residual(q,B_vertex,dx,dy,p)
        q1raw=q+dt*R0
        h1raw=q1raw[...,0]/p.rho
        if np.min(h1raw)>=-tol:
            q1=sanitize(q1raw,p)
            dt1=0.99*max_dt(q1,dx,dy,p)
            if dt<=dt1*(1+2e-13):
                R1=spatial_residual(q1,B_vertex,dx,dy,p)
                q2raw=0.5*q+0.5*(q1+dt*R1)
                if np.min(q2raw[...,0]/p.rho)>=-tol:
                    return sanitize(q2raw,p),dict(dt=dt,retries=retries)
        retries+=1
        if retries>max_retries:
            raise RuntimeError('SSPRK2 positivity/CFL retry limit exceeded')
        dt*=0.5


def run(q0,B_vertex,dx,dy,t_end,p=Params(),output_times=()):
    q=sanitize(q0,p); t=0.0; outs=sorted(set([float(z) for z in output_times if 0<=z<=t_end]+[float(t_end)])); snaps=[];k=0
    if outs and outs[0]<=1e-14: snaps.append((outs[0],q.copy()));k=1
    stats={'steps':0,'retries':0,'min_dt':np.inf,'max_dt':0.0}
    while t<t_end-1e-14:
        cap=t_end-t
        if k<len(outs):cap=min(cap,max(outs[k]-t,0.0))
        if cap<=1e-15:
            snaps.append((outs[k],q.copy()));k+=1;continue
        q,sd=step_ssprk2(q,B_vertex,dx,dy,t,cap,p);t+=sd['dt']
        stats['steps']+=1;stats['retries']+=sd['retries'];stats['min_dt']=min(stats['min_dt'],sd['dt']);stats['max_dt']=max(stats['max_dt'],sd['dt'])
        while k<len(outs) and t>=outs[k]-1e-12:
            snaps.append((outs[k],q.copy()));k+=1
    return snaps,stats


# -----------------------------------------------------------------------------
# Geometry helper: continuous one-cell excavation walls
# -----------------------------------------------------------------------------
def excavation_weight_vertex(x,y,x1,x2,y1,y2,dx):
    """Exact weight used by EXAMPLE_INCL_EXC_2D/create_example.py."""
    wx=np.minimum((x-x1)/dx,(x2-x)/dx)
    wy=np.minimum((y-y1)/dx,(y2-y)/dx)
    return np.clip(wx,0.0,1.0)*np.clip(wy,0.0,1.0)

def inclined_excavation_vertices(xv,yv,angle_deg=10.0,x1=10.0,x2=20.0,yhalf=5.0,depth=2.0,z_ref=10.0,dx=None):
    """Exact continuous one-cell Q1 excavation geometry from the Fortran example."""
    xv=np.asarray(xv,float);yv=np.asarray(yv,float)
    if dx is None: dx=float(np.median(np.diff(xv)))
    Xv,Yv=np.meshgrid(xv,yv)
    B0=z_ref-np.tan(np.deg2rad(angle_deg))*Xv
    W=excavation_weight_vertex(Xv,Yv,x1,x2,-yhalf,yhalf,dx)
    return B0-depth*W

def filled_excavation_ic_from_geometry(B_vertex,p=Params(),angle_deg=10.0,z_ref=10.0,xc=None):
    """Exact benchmark IC h=B0_center-B_center, hence eta=B0 in the wet source."""
    geom=geometry_from_vertices(B_vertex); B=geom['B_cent']; ny,nx=B.shape
    if xc is None: raise ValueError("xc is required")
    xc=np.asarray(xc,float); B0=(z_ref-np.tan(np.deg2rad(angle_deg))*xc)[None,:]
    h=np.maximum(B0-B,0.0)
    return state_from_huv(h,np.zeros_like(h),np.zeros_like(h),p)

