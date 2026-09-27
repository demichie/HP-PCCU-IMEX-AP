"""Cartesian 2-D mass-fraction extension of the HP-PCCU reference core.

Topography is continuous and stored with one shared value per Cartesian face,
normally built from a global Q1 vertex field. Slope and curvature use a local
least-squares fit of cell-centre bed values derived from that geometry.
"""

from dataclasses import dataclass
import numpy as np
from hp_pccu_1d_core import *

def sanitize2d(q, p):
    """Admissibility projection for Q=(M,mx,my,Cs)."""
    q=np.asarray(q,dtype=float).copy()
    q[...,0]=np.maximum(q[...,0],0.0)
    q[...,3]=np.clip(q[...,3],0.0,q[...,0])
    M=q[...,0]; Cs=q[...,3]
    h=(M-Cs)/p.rho_c + Cs/p.rho_s
    dry=(h<=p.dry_h_tol) | (M<=p.M_min)
    for k in range(4):
        q[...,k]=np.where(dry,0.0,q[...,k])
    return q

def primitive2d(q,p):
    q=np.asarray(q,dtype=float)
    M=np.maximum(q[...,0],0.0)
    Cs=np.clip(q[...,3],0.0,M)
    u=desingularized_velocity(M,q[...,1],p.eps_sing)
    v=desingularized_velocity(M,q[...,2],p.eps_sing)
    h=(M-Cs)/p.rho_c + Cs/p.rho_s
    h=np.maximum(h,0.0)
    hsafe=np.maximum(h,max(p.h_min,1e-300))
    rho=np.where(M>p.M_min,M/hsafe,p.rho_c)
    rho=np.maximum(rho,max(p.rho_a+1e-14,1e-14))
    alpha=np.where(h>p.h_min,(Cs/p.rho_s)/hsafe,0.0)
    alpha=np.clip(alpha,0.0,1.0)
    y_s=np.where(M>p.M_min,Cs/np.maximum(M,p.M_min),0.0)
    y_s=np.clip(y_s,0.0,1.0)
    gp=p.g*(rho-p.rho_a)/rho
    Gamma=rho*gp
    speed=np.sqrt(u*u+v*v)
    return dict(M=M,mx=q[...,1],my=q[...,2],Cs=Cs,u=u,v=v,h=h,rho=rho,
                alpha_s=alpha,y_s=y_s,gprime=gp,Gamma=Gamma,speed=speed)

def state2d_from_h_alpha_uv(h,alpha,u,v,p):
    h=np.asarray(h,dtype=float)
    alpha=np.clip(np.asarray(alpha,dtype=float),0.0,1.0)
    u=np.asarray(u,dtype=float); v=np.asarray(v,dtype=float)
    h,alpha,u,v=np.broadcast_arrays(h,alpha,u,v)
    rho=p.rho_c*(1-alpha)+p.rho_s*alpha
    q=np.zeros(h.shape+(4,),dtype=float)
    q[...,0]=rho*np.maximum(h,0.0)
    q[...,1]=q[...,0]*u
    q[...,2]=q[...,0]*v
    q[...,3]=p.rho_s*alpha*np.maximum(h,0.0)
    return sanitize2d(q,p)

def state2d_from_h_massfrac_uv(h,y_s,u,v,p):
    """Build Q=(M,mx,my,Cs) from the canonical solid mass fraction y_s."""
    h=np.maximum(np.asarray(h,dtype=float),0.0)
    y_s=np.clip(np.asarray(y_s,dtype=float),0.0,1.0)
    u=np.asarray(u,dtype=float); v=np.asarray(v,dtype=float)
    h,y_s,u,v=np.broadcast_arrays(h,y_s,u,v)
    inv_rho=(1.0-y_s)/p.rho_c+y_s/p.rho_s
    rho=1.0/np.maximum(inv_rho,1.0e-300)
    q=np.zeros(h.shape+(4,),dtype=float)
    q[...,0]=rho*h
    q[...,1]=q[...,0]*u
    q[...,2]=q[...,0]*v
    q[...,3]=y_s*q[...,0]
    return sanitize2d(q,p)

def _tangential_velocity_slice(qslice,normal,p):
    M=np.maximum(qslice[:,0],0.0)
    mt=qslice[:,2] if normal=='x' else qslice[:,1]
    return desingularized_velocity(M,mt,p.eps_sing)

def _reconstruct_scalar_faces(v,p):
    v=np.asarray(v,dtype=float)
    vLcell,vRcell=reconstruct_array(v,p)
    return np.concatenate([[v[0]],vRcell]), np.concatenate([vLcell,[v[-1]]])

def raw_thickness2d(q,p):
    q=np.asarray(q,float)
    return (q[...,0]-q[...,3])/p.rho_c + q[...,3]/p.rho_s

def mass2d(q,dx,dy):
    return float(np.sum(np.asarray(q)[...,0])*dx*dy)

@dataclass(frozen=True)
class BedGeometry2D:
    """Continuous Cartesian bed with one shared elevation on every face."""
    Bx_face: np.ndarray  # shape (ny,nx+1), vertical faces
    By_face: np.ndarray  # shape (ny+1,nx), horizontal faces

    def __post_init__(self):
        bx=np.asarray(self.Bx_face,dtype=float)
        by=np.asarray(self.By_face,dtype=float)
        if bx.ndim != 2 or by.ndim != 2:
            raise ValueError('face bed arrays must be two-dimensional')
        ny,nxf=bx.shape; nyf,nx=by.shape
        if nxf != nx+1 or nyf != ny+1:
            raise ValueError('incompatible x/y face-array shapes')
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            raise ValueError('non-finite bed face elevation')
        object.__setattr__(self,'Bx_face',bx.copy())
        object.__setattr__(self,'By_face',by.copy())

    @property
    def shape(self):
        return (self.Bx_face.shape[0], self.By_face.shape[1])

    @property
    def B_center_x(self):
        return 0.5*(self.Bx_face[:,:-1]+self.Bx_face[:,1:])

    @property
    def B_center_y(self):
        return 0.5*(self.By_face[:-1,:]+self.By_face[1:,:])

    @property
    def B_center(self):
        bx=self.B_center_x; by=self.B_center_y
        scale=np.maximum(1.0,np.maximum(np.abs(bx),np.abs(by)))
        if np.max(np.abs(bx-by)/scale)>5e-11:
            raise ValueError('x- and y-face geometry imply inconsistent cell-centre beds')
        return 0.5*(bx+by)

    def x_slice(self,j):
        return BedGeometry1D(self.Bx_face[j,:])

    def y_slice(self,i):
        return BedGeometry1D(self.By_face[:,i])


def bed_center_values_2d(B):
    return B.B_center if isinstance(B,BedGeometry2D) else np.asarray(B,float)


@dataclass(frozen=True)
class SlopeGeometry2D:
    G_center: np.ndarray
    Gx_face: np.ndarray
    Gy_face: np.ndarray
    Bx_center: np.ndarray
    By_center: np.ndarray
    Bxx_center: np.ndarray
    Bxy_center: np.ndarray
    Byy_center: np.ndarray

def ls_topography_fit_2d(B_center, dx, dy):
    """Fortran-compatible local quadratic LS slopes and Hessian.

    Pure x/y derivatives use the same five-point quadratic fits as
    ``geometry_2d.f90``.  The mixed derivative uses the tensor product of the
    first-derivative LS kernels, followed by the same copied boundary strip.
    """
    B=np.asarray(B_center,dtype=float)
    if B.ndim != 2:
        raise ValueError('B_center must be two-dimensional')
    ny,nx=B.shape
    bx=np.zeros_like(B); by=np.zeros_like(B)
    bxx=np.zeros_like(B); byy=np.zeros_like(B); bxy=np.zeros_like(B)
    for j in range(ny):
        bx[j,:],bxx[j,:]=ls_quadratic_derivatives_1d(B[j,:],dx)
    for i in range(nx):
        by[:,i],byy[:,i]=ls_quadratic_derivatives_1d(B[:,i],dy)
    if nx >= 5 and ny >= 5:
        c1=np.array([-2.0,-1.0,0.0,1.0,2.0])
        for j in range(2,ny-2):
            for i in range(2,nx-2):
                patch=B[j-2:j+3,i-2:i+3]
                bxy[j,i]=np.sum((c1[:,None]*c1[None,:])*patch)/(100.0*dx*dy)
        # Match the Fortran zero-order extrapolation of the two-cell border.
        for j in range(2,ny-2):
            bxy[j,0]=bxy[j,2]; bxy[j,1]=bxy[j,2]
            bxy[j,-2]=bxy[j,-3]; bxy[j,-1]=bxy[j,-3]
        bxy[0,:]=bxy[2,:]; bxy[1,:]=bxy[2,:]
        bxy[-2,:]=bxy[-3,:]; bxy[-1,:]=bxy[-3,:]
    return bx,by,bxx,bxy,byy

def build_slope_geometry_2d(B, dx, dy):
    """Build G from the LS fit while leaving the continuous HP bed untouched.
    """
    Bc=bed_center_values_2d(B)
    bx,by,bxx,bxy,byy=ls_topography_fit_2d(Bc,dx,dy)
    Gc=1.0/(1.0+bx*bx+by*by)
    ny,nx=Gc.shape
    Gxf=np.empty((ny,nx+1),dtype=float)
    Gyf=np.empty((ny+1,nx),dtype=float)
    Gxf[:,0]=Gc[:,0]; Gxf[:,-1]=Gc[:,-1]
    if nx>1: Gxf[:,1:-1]=0.5*(Gc[:,:-1]+Gc[:,1:])
    Gyf[0,:]=Gc[0,:]; Gyf[-1,:]=Gc[-1,:]
    if ny>1: Gyf[1:-1,:]=0.5*(Gc[:-1,:]+Gc[1:,:])
    return SlopeGeometry2D(Gc,Gxf,Gyf,bx,by,bxx,bxy,byy)

def hydrostatic_residual_2d(h_center, B_center):
    """Current Fortran four-neighbour hydrostatic residual."""
    h=np.asarray(h_center,dtype=float); B=np.asarray(B_center,dtype=float)
    if h.shape != B.shape or h.ndim != 2:
        raise ValueError('h_center and B_center must be matching 2-D arrays')
    eta=h+B; ny,nx=h.shape; r=np.zeros_like(h); eps=np.finfo(float).eps
    for j in range(ny):
        for i in range(nx):
            num=0.0; den=0.0
            for jj,ii in ((j,i-1),(j,i+1),(j-1,i),(j+1,i)):
                if 0<=jj<ny and 0<=ii<nx:
                    num += abs(eta[jj,ii]-eta[j,i])
                    den += abs(h[jj,ii]-h[j,i]) + abs(B[jj,ii]-B[j,i])
            scale=128.0*eps*max(1.0,abs(eta[j,i]),abs(h[j,i]),abs(B[j,i]))
            r[j,i]=min(1.0,max(0.0,num/den)) if den>scale else 0.0
    return r

def topographic_relief_ratio_2d(h_center, B):
    """Within-cell four-face relief divided by local thickness."""
    h=np.asarray(h_center,dtype=float)
    if not isinstance(B,BedGeometry2D):
        raise ValueError('2-D dry-safe relief requires BedGeometry2D shared-face geometry')
    vals=np.stack((B.Bx_face[:,:-1],B.Bx_face[:,1:],
                   B.By_face[:-1,:],B.By_face[1:,:]),axis=0)
    relief=np.max(vals,axis=0)-np.min(vals,axis=0)
    return relief/np.maximum(h,P0.dry_h_tol)

def continuous_q1_bed_geometry(B_vertex):
    """Build the shared HP face geometry from a global Q1 vertex field."""
    V=np.asarray(B_vertex,dtype=float)
    if V.ndim != 2 or min(V.shape) < 2:
        raise ValueError('B_vertex must have shape (ny+1,nx+1)')
    Bx=0.5*(V[:-1,:]+V[1:,:])
    By=0.5*(V[:,:-1]+V[:,1:])
    return BedGeometry2D(Bx,By)


def _x_bed_slice(B,j):
    return B.x_slice(j) if isinstance(B,BedGeometry2D) else np.asarray(B[j,:],float)


def _y_bed_slice(B,i):
    return B.y_slice(i) if isinstance(B,BedGeometry2D) else np.asarray(B[:,i],float)


def _tangential_flux_from_diag(qslice,normal,p,diag):
    """Baseline CU flux for tangential momentum; no alternative face closure."""
    vt=_tangential_velocity_slice(qslice,normal,p)
    vtL,vtR=_reconstruct_scalar_faces(vt,p)
    ML=diag['qL_face'][:,0]; MR=diag['qR_face'][:,0]
    mtL=ML*vtL; mtR=MR*vtR
    am=np.asarray(diag['a_minus']); ap=np.asarray(diag['a_plus'])
    den=ap-am
    ds=np.where(den>p.speed_eps,den,1.0)
    fL=np.asarray(diag['uL_face'])*mtL
    fR=np.asarray(diag['uR_face'])*mtR
    Ht=(ap*fL-am*fR+ap*am*(mtR-mtL))/ds
    deg=den<=p.speed_eps
    Ht[deg]=0.5*(fL[deg]+fR[deg])
    return Ht


def _evaluate_bed2d_stage(B, t, q, p):
    """Resolve bed geometry independently at every RK stage."""
    Bstage = B(float(t), q, p) if callable(B) else B
    ny, nx = q.shape[:2]
    if isinstance(Bstage, BedGeometry2D):
        if Bstage.shape != (ny, nx):
            raise ValueError('BedGeometry2D shape does not match state')
    else:
        Barr=np.asarray(Bstage,dtype=float)
        if Barr.shape != (ny,nx):
            raise ValueError('2-D cell-centred bed shape does not match state')
        Bstage=Barr
    return Bstage


def prepare_stage_cache2d(q, B, t, p, q_is_sanitized=False, dx=None, dy=None,
                          slope_geometry=None):
    """Build all Cartesian HP-PCCU face data once for one RK stage."""
    p.checked()
    qstage=np.asarray(q,dtype=float) if q_is_sanitized else sanitize2d(q,p)
    if qstage.ndim != 3 or qstage.shape[-1] != 4:
        raise ValueError('2-D state must have shape (ny,nx,4)')
    ny,nx,_=qstage.shape
    Bstage=_evaluate_bed2d_stage(B,t,qstage,p)
    prim_stage=primitive2d(qstage,p)
    Bcenter=bed_center_values_2d(Bstage)
    hydro_residual=hydrostatic_residual_2d(prim_stage['h'],Bcenter)
    if isinstance(Bstage,BedGeometry2D):
        vals=np.stack((Bstage.Bx_face[:,:-1],Bstage.Bx_face[:,1:],
                       Bstage.By_face[:-1,:],Bstage.By_face[1:,:]),axis=0)
        relief=np.max(vals,axis=0)-np.min(vals,axis=0)
    else:
        # Cell-centred fallback used only by lightweight tests.  Approximate
        # four-face relief from directional reconstructed bed endpoints.
        BL=np.empty_like(Bcenter); BR=np.empty_like(Bcenter)
        BS=np.empty_like(Bcenter); BN=np.empty_like(Bcenter)
        for jj in range(ny): BL[jj],BR[jj]=reconstruct_bed(Bcenter[jj],p)
        for ii in range(nx): BS[:,ii],BN[:,ii]=reconstruct_bed(Bcenter[:,ii],p)
        vals=np.stack((BL,BR,BS,BN),axis=0)
        relief=np.max(vals,axis=0)-np.min(vals,axis=0)
    relief_ratio=relief/np.maximum(prim_stage['h'],p.dry_h_tol)
    need_ls_geometry=p.slope_correction or p.curvature_term
    if need_ls_geometry:
        if slope_geometry is None:
            if dx is None or dy is None:
                raise ValueError('slope/curvature correction requires dx and dy')
            slope_geometry=build_slope_geometry_2d(Bstage,dx,dy)
    else:
        slope_geometry=None

    x_entries=[]
    for j in range(ny):
        qn=np.stack([qstage[j,:,0],qstage[j,:,1],qstage[j,:,3]],axis=-1)
        B1=_x_bed_slice(Bstage,j)
        sg1=None if slope_geometry is None else SlopeGeometry1D(
            slope_geometry.G_center[j,:],slope_geometry.Gx_face[j,:],
            slope_geometry.Bx_center[j,:],slope_geometry.Bxx_center[j,:])
        c1=prepare_stage_face_cache(qn,B1,t,p,transmissive_boundary,
                                    q_is_sanitized=True,slope_geometry=sg1,
                                    hydrostatic_residual=hydro_residual[j,:],
                                    topographic_relief_ratio=relief_ratio[j,:])
        x_entries.append(dict(normal_state=qn,bed=B1,cache1d=c1,slope_geometry=sg1))

    y_entries=[]
    for i in range(nx):
        qn=np.stack([qstage[:,i,0],qstage[:,i,2],qstage[:,i,3]],axis=-1)
        B1=_y_bed_slice(Bstage,i)
        sg1=None if slope_geometry is None else SlopeGeometry1D(
            slope_geometry.G_center[:,i],slope_geometry.Gy_face[:,i],
            slope_geometry.By_center[:,i],slope_geometry.Byy_center[:,i])
        c1=prepare_stage_face_cache(qn,B1,t,p,transmissive_boundary,
                                    q_is_sanitized=True,slope_geometry=sg1,
                                    hydrostatic_residual=hydro_residual[:,i],
                                    topographic_relief_ratio=relief_ratio[:,i])
        y_entries.append(dict(normal_state=qn,bed=B1,cache1d=c1,slope_geometry=sg1))

    return dict(q=qstage,B=Bstage,t=float(t),x=x_entries,y=y_entries,
                slope_geometry=slope_geometry,
                hydrostatic_residual=hydro_residual,
                topographic_relief_ratio=relief_ratio)


def rhs_direction_slice(qslice, Bslice, d, t, p, normal,
                        return_diag=False, stage_entry=None,
                        qslice_is_sanitized=False):
    qslice=np.asarray(qslice,dtype=float) if qslice_is_sanitized else sanitize2d(qslice,p)
    if stage_entry is None:
        qn=np.stack([qslice[:,0],
                     qslice[:,1] if normal=='x' else qslice[:,2],
                     qslice[:,3]],axis=-1)
        B1=Bslice if isinstance(Bslice,BedGeometry1D) else np.asarray(Bslice,float)
        c1=None
    else:
        qn=stage_entry['normal_state']; B1=stage_entry['bed']; c1=stage_entry['cache1d']

    Rn,diag=rhs_spatial(qn,B1,d,t,p,transmissive_boundary,
                        return_diagnostics=True,stage_cache=c1,
                        include_curvature_source=False)
    Ht=_tangential_flux_from_diag(qslice,normal,p,diag)
    Rt=-(Ht[1:]-Ht[:-1])/d
    R=np.zeros_like(qslice)
    R[:,0]=Rn[:,0]
    if normal=='x':
        R[:,1]=Rn[:,1]; R[:,2]=Rt
    else:
        R[:,2]=Rn[:,1]; R[:,1]=Rt
    R[:,3]=Rn[:,2]
    return (R,diag,Ht) if return_diag else R


def rhs2d(q, B, dx, dy, t, p, return_diagnostics=False, stage_cache=None):
    """Unsplit Cartesian baseline HP-PCCU residual."""
    cache=(prepare_stage_cache2d(q,B,t,p,dx=dx,dy=dy)
           if stage_cache is None else stage_cache)
    qstage=cache['q']; ny,nx,_=qstage.shape
    R=np.zeros_like(qstage); max_face_normal=0.0

    for j,entry in enumerate(cache['x']):
        r,dg,_=rhs_direction_slice(qstage[j,:,:],entry['bed'],dx,t,p,'x',True,
                                   stage_entry=entry,qslice_is_sanitized=True)
        R[j,:,:]+=r
        max_face_normal=max(max_face_normal,float(np.max(np.abs(dg['uL_face']))),
                            float(np.max(np.abs(dg['uR_face']))))
    for i,entry in enumerate(cache['y']):
        r,dg,_=rhs_direction_slice(qstage[:,i,:],entry['bed'],dy,t,p,'y',True,
                                   stage_entry=entry,qslice_is_sanitized=True)
        R[:,i,:]+=r
        max_face_normal=max(max_face_normal,float(np.max(np.abs(dg['uL_face']))),
                            float(np.max(np.abs(dg['uR_face']))))

    curvature_accel=np.zeros(qstage.shape[:2],dtype=float)
    curvature_source_x=np.zeros_like(curvature_accel)
    curvature_source_y=np.zeros_like(curvature_accel)
    if p.curvature_term:
        sg=cache.get('slope_geometry')
        if sg is None:
            raise ValueError('curvature_term requires LS slope/curvature geometry')
        f=primitive2d(qstage,p)
        curvature_accel=(sg.Bxx_center*f['u']**2
                         +2.0*sg.Bxy_center*f['u']*f['v']
                         +sg.Byy_center*f['v']**2)
        Gc=sg.G_center if p.slope_correction else 1.0
        curvature_source_x=-Gc*f['M']*curvature_accel*sg.Bx_center
        curvature_source_y=-Gc*f['M']*curvature_accel*sg.By_center
        R[...,1]+=curvature_source_x; R[...,2]+=curvature_source_y

    if return_diagnostics:
        return R,dict(max_abs_normal_face_velocity=max_face_normal,
                      curvature_accel=curvature_accel,
                      curvature_source_x=curvature_source_x,
                      curvature_source_y=curvature_source_y,
                      stage_cache_reused=stage_cache is not None)
    return R


def max_dt2d(q, B, dx, dy, t, p, stage_cache=None):
    cache=(prepare_stage_cache2d(q,B,t,p,dx=dx,dy=dy)
           if stage_cache is None else stage_cache)
    dtx=np.inf; dty=np.inf; maxfac=0.0; spmax=0.0
    for entry in cache['x']:
        dt,dg=max_dt(entry['normal_state'],entry['bed'],dx,t,p,
                     transmissive_boundary,True,stage_cache=entry['cache1d'])
        dtx=min(dtx,float(dt)); maxfac=max(maxfac,float(dg['max_abs_u_face']))
        spmax=max(spmax,float(dg['speed_max']))
    for entry in cache['y']:
        dt,dg=max_dt(entry['normal_state'],entry['bed'],dy,t,p,
                     transmissive_boundary,True,stage_cache=entry['cache1d'])
        dty=min(dty,float(dt)); maxfac=max(maxfac,float(dg['max_abs_u_face']))
        spmax=max(spmax,float(dg['speed_max']))
    dt=1.0/(1.0/max(dtx,1e-300)+1.0/max(dty,1e-300)) if np.isfinite(dtx) and np.isfinite(dty) else min(dtx,dty)
    return dt,dict(dt_x=dtx,dt_y=dty,max_abs_normal_face_velocity=maxfac,
                   speed_max=spmax,stage_cache_reused=stage_cache is not None)


def step_ssprk2_2d(q, B, dx, dy, t, dt_cap, p, max_retries=12):
    q=sanitize2d(q,p)
    cache0=prepare_stage_cache2d(q,B,t,p,q_is_sanitized=True,dx=dx,dy=dy)
    dt0,d0=max_dt2d(q,B,dx,dy,t,p,stage_cache=cache0)
    dt=min(float(dt_cap),float(dt0)); retries=0
    tol_h=5e-13*max(1.0,float(np.max(primitive2d(q,p)['h'])))
    R0,rd0=rhs2d(q,B,dx,dy,t,p,True,stage_cache=cache0)
    while True:
        q1raw=q+dt*R0; min_h1=float(np.min(raw_thickness2d(q1raw,p))); q1=sanitize2d(q1raw,p)
        cache1=prepare_stage_cache2d(q1,B,t+dt,p,q_is_sanitized=True,dx=dx,dy=dy)
        dt1,d1=max_dt2d(q1,B,dx,dy,t+dt,p,stage_cache=cache1)
        if dt <= float(dt1)*(1.0+2e-13) and min_h1 >= -tol_h:
            R1,rd1=rhs2d(q1,B,dx,dy,t+dt,p,True,stage_cache=cache1)
            q2raw=0.5*q+0.5*(q1+dt*R1); min_h2=float(np.min(raw_thickness2d(q2raw,p)))
            if min_h2 >= -tol_h:
                q2=sanitize2d(q2raw,p)
                return q2,dict(dt=dt,dt_cfl_stage0=float(dt0),dt_cfl_stage1=float(dt1),
                               retries=retries,min_raw_h=min(min_h1,min_h2),
                               min_raw_M=min(float(np.min(q1raw[...,0])),float(np.min(q2raw[...,0]))),
                               max_abs_normal_face_velocity=max(d0['max_abs_normal_face_velocity'],
                                                               d1['max_abs_normal_face_velocity'],
                                                               rd0['max_abs_normal_face_velocity'],
                                                               rd1['max_abs_normal_face_velocity']),
                               speed_max=max(d0['speed_max'],d1['speed_max']))
        retries+=1
        if retries>max_retries:
            raise RuntimeError('2D SSPRK2 stagewise CFL/positivity check failed')
        dt_new=min(0.8*dt,0.95*float(dt1))
        if min_h1 < -tol_h: dt_new=min(dt_new,0.5*dt)
        if not (dt_new>1e-15 and dt_new<dt): dt_new=0.5*dt
        dt=dt_new

