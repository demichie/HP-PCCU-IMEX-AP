"""HP-PCCU 1-D reference core.

The module implements the hydrostatic-path PCCU spatial operator used by the
reproducibility benchmarks. The bed representation is continuous: one unique
elevation is stored at each face. Slope correction and curvature are auxiliary
LS-derived geometry and do not redefine the HP face elevations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SolverParams:
    rho_c: float = 1000.0
    rho_s: float = 2500.0
    rho_a: float = 0.0
    g: float = 9.81

    cfl: float = 0.24
    limiter: int = 3
    theta: float = 1.3
    reconstr_coeff: float = 1.0
    n_quad_path: int = 3
    h_min: float = 0.0
    M_min: float = 1.0e-12
    speed_eps: float = 1.0e-14
    eps_sing: float = 1.0e-10

    reconstruct_eta: bool = True
    slope_correction: bool = False
    curvature_term: bool = False

    dry_h_tol: float = 1.0e-10
    bed_step_tol: float = 1.0e-12
    hp_dynamic_residual_threshold: float = 0.90

    zero_transport_at_rest: bool = True
    rest_flux_tol: float = 1.0e-14

    def checked(self):
        if self.n_quad_path not in (1, 2, 3):
            raise ValueError("n_quad_path must be 1, 2, or 3")
        if self.eps_sing <= 0.0:
            raise ValueError("eps_sing must be positive")
        if self.rest_flux_tol < 0.0:
            raise ValueError("rest_flux_tol must be non-negative")
        return self


@dataclass(frozen=True)
class BedGeometry1D:
    """Continuous 1-D bed with one shared elevation per face."""
    B_face: np.ndarray

    def __post_init__(self):
        bf=np.asarray(self.B_face,dtype=float)
        if bf.ndim != 1 or bf.size < 2:
            raise ValueError('B_face must be a one-dimensional array with at least two faces')
        if not np.all(np.isfinite(bf)):
            raise ValueError('bed face elevations must be finite')
        object.__setattr__(self,'B_face',bf.copy())

    @property
    def ncell(self):
        return self.B_face.size-1

    @property
    def B_left_cell(self):
        return self.B_face[:-1]

    @property
    def B_right_cell(self):
        return self.B_face[1:]

    @property
    def B_center(self):
        return 0.5*(self.B_left_cell+self.B_right_cell)


@dataclass(frozen=True)
class SlopeGeometry1D:
    """Stage-local LS metric data; independent of the HP face bed geometry."""
    G_center: np.ndarray
    G_face: np.ndarray
    Bx_center: np.ndarray | None = None
    Bxx_center: np.ndarray | None = None

    def __post_init__(self):
        gc=np.asarray(self.G_center,dtype=float)
        gf=np.asarray(self.G_face,dtype=float)
        if gc.ndim != 1 or gf.ndim != 1 or gf.size != gc.size+1:
            raise ValueError('SlopeGeometry1D requires G_center[n] and G_face[n+1]')
        if not (np.all(np.isfinite(gc)) and np.all(np.isfinite(gf))):
            raise ValueError('non-finite slope-correction coefficient')
        if np.any(gc <= 0.0) or np.any(gf <= 0.0):
            raise ValueError('slope-correction coefficient must be positive')
        object.__setattr__(self,'G_center',gc.copy())
        object.__setattr__(self,'G_face',gf.copy())
        for name in ('Bx_center','Bxx_center'):
            val=getattr(self,name)
            if val is not None:
                arr=np.asarray(val,dtype=float)
                if arr.shape != gc.shape:
                    raise ValueError(f'{name} must have shape {gc.shape}')
                object.__setattr__(self,name,arr.copy())


def continuous_bed_geometry(B_face):
    return BedGeometry1D(np.asarray(B_face,dtype=float))


def bed_center_values(B):
    if isinstance(B,BedGeometry1D):
        return B.B_center
    return np.asarray(B,dtype=float)


P0 = SolverParams().checked()

def ls_quadratic_derivatives_1d(B_center, dx):
    """Quadratic LS slope/curvature used by the current Fortran geometry.

    Interior cells use the five-point quadratic least-squares formulas

      Bx  = (-2 B[-2]-B[-1]+B[+1]+2 B[+2])/(10 dx),
      Bxx = ( 2 B[-2]-B[-1]-2B[0]-B[+1]+2B[+2])/(7 dx^2).

    As in ``geometry_2d.f90``, the two-cell boundary strip copies the nearest
    valid interior value.  Very short arrays fall back to a direct global
    quadratic least-squares fit.
    """
    B=np.asarray(B_center,dtype=float)
    if B.ndim != 1:
        raise ValueError('B_center must be one-dimensional')
    n=B.size
    if n == 0:
        return B.copy(), B.copy()
    if not (dx > 0.0):
        raise ValueError('dx must be positive')
    bx=np.zeros(n,dtype=float); bxx=np.zeros(n,dtype=float)
    if n >= 5:
        c1=np.array([-2.0,-1.0,0.0,1.0,2.0])
        c2=np.array([2.0,-1.0,-2.0,-1.0,2.0])
        for i in range(2,n-2):
            st=B[i-2:i+3]
            bx[i]=np.dot(c1,st)/(10.0*dx)
            bxx[i]=np.dot(c2,st)/(7.0*dx*dx)
        bx[:2]=bx[2]; bx[-2:]=bx[-3]
        bxx[:2]=bxx[2]; bxx[-2:]=bxx[-3]
    elif n >= 3:
        x=(np.arange(n)-0.5*(n-1))*dx
        A=np.stack([np.ones(n),x,x*x],axis=1)
        coef=np.linalg.lstsq(A,B,rcond=None)[0]
        # Evaluate derivatives of the fitted polynomial at each cell centre.
        bx[:]=coef[1]+2.0*coef[2]*x
        bxx[:]=2.0*coef[2]
    elif n == 2:
        bx[:]=(B[1]-B[0])/dx
    return bx,bxx

def build_slope_geometry_1d(B, dx):
    """Build cell and shared-face G from the LS topographic slope.

    The bed values used by HP reconstruction are untouched.  Only the
    auxiliary metric coefficient G is derived here.
    """
    Bc = B.B_center if isinstance(B, BedGeometry1D) else np.asarray(B,dtype=float)
    bx,bxx=ls_quadratic_derivatives_1d(Bc,dx)
    gc=1.0/(1.0+bx*bx)
    gf=np.empty(gc.size+1,dtype=float)
    if gc.size:
        gf[0]=gc[0]; gf[-1]=gc[-1]
        if gc.size > 1:
            gf[1:-1]=0.5*(gc[:-1]+gc[1:])
    return SlopeGeometry1D(gc,gf,bx,bxx)

def fortran_eps_sing_1d(dx):
    return min(float(dx)**4, 1.0e-10)

def sanitize(q, p):
    """Project conservative states onto the admissible set.

    The wet/dry interface logic classifies a state using the physical thickness
    h <= dry_h_tol.  Use the *same* criterion here, rather than a separate mass
    threshold, so a state cannot be dry for the interface solver while still
    retaining finite momentum.  States below the dry tolerance are collapsed to
    the exact dry state q=0; the discarded mass is at most the prescribed dry
    tolerance and prevents residual momentum from producing enormous near-front
    velocities.
    """
    q = np.asarray(q, dtype=float).copy()
    q[..., 0] = np.maximum(q[..., 0], 0.0)
    q[..., 2] = np.clip(q[..., 2], 0.0, q[..., 0])

    M = q[..., 0]
    Cs = q[..., 2]
    h = (M - Cs) / p.rho_c + Cs / p.rho_s
    dry = (h <= p.dry_h_tol) | (M <= p.M_min)

    q[..., 0] = np.where(dry, 0.0, q[..., 0])
    q[..., 1] = np.where(dry, 0.0, q[..., 1])
    q[..., 2] = np.where(dry, 0.0, q[..., 2])
    return q

def desingularized_velocity(M, momentum, eps_sing):
    """Fortran-like regularized conversion from momentum to velocity.

    Matches the _dev formula:
      u = m/M,                                      M > eps_sing
      u = sqrt(2)*M*m/sqrt(M**4 + eps_sing**4),   otherwise

    The two branches are continuous at M=eps_sing and u -> 0 as M -> 0
    for finite momentum.
    """
    M = np.maximum(np.asarray(M, dtype=float), 0.0)
    momentum = np.asarray(momentum, dtype=float)
    M, momentum = np.broadcast_arrays(M, momentum)
    eps = float(eps_sing)

    u_regular = np.zeros_like(M, dtype=float)
    np.divide(momentum, M, out=u_regular, where=(M > eps))

    denom = np.sqrt(M**4 + eps**4)
    u_small = np.sqrt(2.0) * M * momentum / denom

    return np.where(M > eps, u_regular, u_small)

def primitive(q, p):
    # q=(M,m,Cs), with m=M*u
    q = np.asarray(q, dtype=float)
    M = np.maximum(q[..., 0], 0.0)
    Cs = np.clip(q[..., 2], 0.0, M)

    # Match the near-dry q -> qp velocity conversion used in _dev.
    u = desingularized_velocity(M, q[..., 1], p.eps_sing)

    h = (M - Cs) / p.rho_c + Cs / p.rho_s
    h = np.maximum(h, 0.0)
    h_safe = np.maximum(h, max(p.h_min, 1.0e-300))

    # A dry endpoint has undefined mixture composition. rho_c is used only as
    # a benign endpoint value. Wet-to-dry Gauss points lie inside the wet path.
    rho = np.where(M > p.M_min, M / h_safe, p.rho_c)
    rho = np.maximum(rho, max(p.rho_a + 1.0e-14, 1.0e-14))

    alpha_s = np.where(h > p.h_min, (Cs / p.rho_s) / h_safe, 0.0)
    alpha_s = np.clip(alpha_s, 0.0, 1.0)
    y_s = np.where(M > p.M_min, Cs / np.maximum(M, p.M_min), 0.0)

    gprime = p.g * (rho - p.rho_a) / rho
    Gamma = rho * gprime
    return M, u, Cs, h, rho, alpha_s, y_s, gprime, Gamma

def state_from_h_alpha_u(h, alpha_s, u, p):
    h = max(float(h), 0.0)
    alpha_s = float(np.clip(alpha_s, 0.0, 1.0))
    rho = p.rho_c * (1.0 - alpha_s) + p.rho_s * alpha_s
    M = rho * h
    Cs = p.rho_s * alpha_s * h
    return np.array([M, M * float(u), Cs], dtype=float)

def pure_liquid_state(h, u, p):
    return state_from_h_alpha_u(h, 0.0, u, p)

def hydrostatic_pressure(q, p):
    _, _, _, h, _, _, _, _, Gamma = primitive(q, p)
    return 0.5 * Gamma * h**2

def conservative_flux(q, p, u_adv=None, primitive_values=None):
    """Inertial conservative flux using a separately reconstructed velocity.

    This intentionally mirrors IMEX_SfloW2D_dev: the conservative interface
    state carries M and m=rho*hu, while the advective multiplier is the
    independently reconstructed u.  If u_adv is omitted, the consistent
    velocity m/M is used (useful for standalone algebraic tests).
    """
    vals = primitive(q, p) if primitive_values is None else primitive_values
    M, u_cons, Cs, h, rho, alpha_s, y_s, gp, Gamma = vals
    u_flux = u_cons if u_adv is None else np.asarray(u_adv, dtype=float)

    f = np.zeros_like(q, dtype=float)
    f[..., 0] = u_flux * M
    f[..., 1] = u_flux * q[..., 1]
    f[..., 2] = u_flux * Cs
    return f

def acoustic_speed_exact(q, p):
    """Exact acoustic speed of the homogeneous (M,m,Cs) mixture subsystem.

    Composition is a material invariant along the two acoustic families, so the
    eigenvalues are u-c, u, u+c with c^2=gprime*h.  This differs from a partial
    derivative of pressure at fixed conservative Cs, which changes composition.
    """
    M, u, Cs, h, rho, alpha_s, y_s, gp, Gamma = primitive(q, p)
    return np.sqrt(np.maximum(gp * h, 0.0))

def wave_speeds(qL, qR, p, uL_adv=None, uR_adv=None,
                primitive_L=None, primitive_R=None, grav_coeff=1.0):
    """One-sided CU bounds from the exact three-field eigenvalues.

    The auxiliary reconstructed velocity is retained, as in the Fortran-like
    interface kinematics, while c=sqrt(gprime*h) is evaluated from each
    reconstructed conservative face state.
    """
    def one_side(q, u_adv, vals):
        vals = primitive(q, p) if vals is None else vals
        M, u_cons, Cs, h, rho, alpha_s, y_s, gp, Gamma = vals
        u_wave = u_cons if u_adv is None else np.asarray(u_adv, dtype=float)
        c = np.sqrt(np.maximum(np.asarray(grav_coeff,dtype=float) * gp * h, 0.0))
        return u_wave - c, u_wave + c

    lmL, lpL = one_side(qL, uL_adv, primitive_L)
    lmR, lpR = one_side(qR, uR_adv, primitive_R)
    a_minus = np.minimum(np.minimum(lmL, lmR), 0.0)
    a_plus = np.maximum(np.maximum(lpL, lpR), 0.0)
    return a_minus, a_plus

def minmod(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a * b > 0.0,
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)),
                    0.0)

def reconstruct_array(v, p):
    v = np.asarray(v, dtype=float)
    slope = np.zeros_like(v)

    if p.limiter != 0 and len(v) >= 3:
        dm = v[1:-1] - v[:-2]
        dp = v[2:] - v[1:-1]
        dc = 0.5 * (v[2:] - v[:-2])

        if p.limiter == 3:
            slope[1:-1] = minmod(dc, p.theta * minmod(dm, dp))
        elif p.limiter == 5:
            slope[1:-1] = dc
        else:
            slope[1:-1] = minmod(dm, dp)

    delta = 0.5 * p.reconstr_coeff * slope
    return v - delta, v + delta

def state_from_h_alpha_hu(h, alpha_s, hu, p):
    """Build q=(M,m,Cs) from reconstructed h, alpha_s and h*u.

    The reconstructed conservative momentum is m=rho*(hu).  The separately
    reconstructed velocity is *not* used to rebuild m, matching the _dev split
    between q_interface and the auxiliary u component of qp_interface.
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    alpha_s = np.clip(np.asarray(alpha_s, dtype=float), 0.0, 1.0)
    hu = np.asarray(hu, dtype=float)

    dry = h <= p.dry_h_tol
    hu = np.where(dry, 0.0, hu)
    rho = p.rho_c * (1.0 - alpha_s) + p.rho_s * alpha_s

    q = np.zeros(h.shape + (3,), dtype=float)
    q[..., 0] = rho * h
    q[..., 1] = rho * hu
    q[..., 2] = p.rho_s * alpha_s * h
    return sanitize(q, p)

def state_from_h_massfrac_hu(h, y_s, hu, p):
    """Build q=(M,m,Cs) from h, solid mass fraction y_s=Cs/M, and h*u.

    Mass fraction is the canonical reconstructed composition coordinate.
    For a two-constituent mixture with constant intrinsic densities,
        1/rho = (1-y_s)/rho_c + y_s/rho_s.
    The evolved conservative state remains q=(M,m,Cs).
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    y_s = np.clip(np.asarray(y_s, dtype=float), 0.0, 1.0)
    hu = np.asarray(hu, dtype=float)

    dry = h <= p.dry_h_tol
    hu = np.where(dry, 0.0, hu)
    inv_rho = (1.0-y_s)/p.rho_c + y_s/p.rho_s
    rho = 1.0 / np.maximum(inv_rho, 1.0e-300)

    q = np.zeros(h.shape + (3,), dtype=float)
    q[..., 0] = rho*h
    q[..., 1] = rho*hu
    q[..., 2] = y_s*q[..., 0]
    return sanitize(q,p)

def reconstruct_state(q, p, return_physical=False, primitive_values=None):
    """Fortran-like physical reconstruction of h, hu, u and solid mass fraction.

    h and hu define the conservative face state.  u is reconstructed
    independently with the same limiter used for hu and is carried as an
    auxiliary face variable for fluxes and characteristic speeds.  Eta is
    reconstructed separately later because it also depends on the bed.

    ``primitive_values`` is an optional internal optimization hook.  When the
    caller has already converted exactly the same conservative state, reuse
    those values rather than repeating ``primitive``/desingularization.
    """
    vals = primitive(q, p) if primitive_values is None else primitive_values
    h = vals[3]
    u = vals[1]
    y_s = vals[6]
    hu = h * u

    hL_raw, hR_raw = reconstruct_array(h, p)
    huL, huR = reconstruct_array(hu, p)
    # Same reconstruction operator/limiter for u and hu, as in _dev.
    uL, uR = reconstruct_array(u, p)
    yL, yR = reconstruct_array(y_s, p)

    hL = np.maximum(hL_raw, 0.0)
    hR = np.maximum(hR_raw, 0.0)
    qL = state_from_h_massfrac_hu(hL, yL, huL, p)
    qR = state_from_h_massfrac_hu(hR, yR, huR, p)

    if return_physical:
        return qL, qR, hL, hR, hL_raw, hR_raw, huL, huR, uL, uR
    return qL, qR

def reconstruct_bed(B, p):
    return reconstruct_array(bed_center_values(B), p)

def quadrature_rule(n):
    if n == 1:
        return np.array([0.5]), np.array([1.0])
    if n == 2:
        a = 1.0 / np.sqrt(3.0)
        return 0.5 * np.array([1.0 - a, 1.0 + a]), 0.5 * np.array([1.0, 1.0])
    a = np.sqrt(3.0 / 5.0)
    return (
        0.5 * np.array([1.0 - a, 1.0, 1.0 + a]),
        0.5 * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
    )

def hydrostatic_path_values(q, p):
    """Return only ``h`` and ``Gamma`` needed by hydrostatic paths.

    This deliberately reproduces the corresponding arithmetic in ``primitive``
    but skips the desingularized velocity and all composition variables that the
    hydrostatic path does not use.  Keeping the same operation sequence for h,
    rho, gprime and Gamma preserves the existing floating-point path values.
    """
    q = np.asarray(q, dtype=float)
    M = np.maximum(q[..., 0], 0.0)
    Cs = np.clip(q[..., 2], 0.0, M)

    h = (M - Cs) / p.rho_c + Cs / p.rho_s
    h = np.maximum(h, 0.0)
    h_safe = np.maximum(h, max(p.h_min, 1.0e-300))

    rho = np.where(M > p.M_min, M / h_safe, p.rho_c)
    rho = np.maximum(rho, max(p.rho_a + 1.0e-14, 1.0e-14))
    gprime = p.g * (rho - p.rho_a) / rho
    Gamma = rho * gprime
    return h, Gamma

def path_endpoint_variables(qL, qR, p, etaL=None, etaR=None, BL=None, BR=None,
                            primitive_L=None, primitive_R=None):
    if primitive_L is None:
        hL, GammaL = hydrostatic_path_values(qL, p)
    else:
        hL, GammaL = primitive_L[3], primitive_L[8]
    if primitive_R is None:
        hR, GammaR = hydrostatic_path_values(qR, p)
    else:
        hR, GammaR = primitive_R[3], primitive_R[8]
    if etaL is None or etaR is None:
        if BL is None or BR is None:
            raise ValueError("provide eta endpoints or bed endpoints")
        etaL = hL + np.asarray(BL)
        etaR = hR + np.asarray(BR)
    return hL, hR, GammaL, GammaR, np.asarray(etaL), np.asarray(etaR)

def path_hydrostatic_source(qL, qR, etaL, etaR, p,
                              primitive_L=None, primitive_R=None,
                              grav_coeff=1.0, grav_coeff_right=None):
    # Integral of -(Gamma*h*deta + 0.5*h^2*dGamma).
    qL = np.asarray(qL)
    out = np.zeros_like(qL, dtype=float)

    hL, hR, GammaL, GammaR, etaL, etaR = path_endpoint_variables(
        qL, qR, p, etaL=etaL, etaR=etaR,
        primitive_L=primitive_L, primitive_R=primitive_R
    )
    dEta = etaR - etaL
    dGamma = GammaR - GammaL
    GL = np.asarray(grav_coeff,dtype=float)
    GR = GL if grav_coeff_right is None else np.asarray(grav_coeff_right,dtype=float)

    s_pts, w_pts = quadrature_rule(p.n_quad_path)
    integ = np.zeros_like(np.asarray(dEta), dtype=float)
    for s, w in zip(s_pts, w_pts):
        hs = hL + s * (hR - hL)
        Gammas = GammaL + s * dGamma
        Gs = GL + s * (GR - GL)
        integ += w * Gs * (-Gammas * hs * dEta - 0.5 * hs**2 * dGamma)

    out[..., 1] = integ
    return out

def cu_flux_from_states(qL, qR, p, jump_override=None, uL_aux=None, uR_aux=None,
                        primitive_L=None, primitive_R=None, grav_coeff=1.0):
    fL = conservative_flux(qL, p, u_adv=uL_aux, primitive_values=primitive_L)
    fR = conservative_flux(qR, p, u_adv=uR_aux, primitive_values=primitive_R)
    am, ap = wave_speeds(
        qL, qR, p, uL_adv=uL_aux, uR_adv=uR_aux,
        primitive_L=primitive_L, primitive_R=primitive_R,
        grav_coeff=grav_coeff
    )
    den = ap - am
    den_safe = np.where(den > p.speed_eps, den, 1.0)
    jump = qR - qL if jump_override is None else jump_override
    H = (ap[:, None] * fL - am[:, None] * fR + ap[:, None] * am[:, None] * jump) / den_safe[:, None]
    deg = den <= p.speed_eps
    if np.any(deg):
        H[deg] = 0.5 * (fL[deg] + fR[deg])
    return H, fL, fR, am, ap

def pccu_effective_face_values(H, Bpsi, a_minus, a_plus, speed_eps):
    """Return the two oriented effective face values G_L and G_R.

    Cell L receives -G_L/dx from its right face and cell R receives +G_R/dx
    from its left face.  For the ordinary PCCU discretization

        G_L = H + a-/(a+-a-) Bpsi,
        G_R = H + a+/(a+-a-) Bpsi,

    hence G_R-G_L = Bpsi exactly (up to roundoff).  This identity is the local
    path-consistency diagnostic used below.
    """
    den = a_plus - a_minus
    den_safe = np.where(den > speed_eps, den, 1.0)
    # GL is the effective value seen as the right boundary of the left cell;
    # GR is the value seen as the left boundary of the right cell.
    GL = H + (a_minus / den_safe)[:, None] * Bpsi
    GR = H + (a_plus / den_safe)[:, None] * Bpsi
    deg = den <= speed_eps
    if np.any(deg):
        GL[deg] = H[deg]
        GR[deg] = H[deg]
    return GL, GR

def local_hydrostatic_residual_1d(h_center, B_center):
    """Fortran-aligned dimensionless hydrostatic residual in one dimension.

    This is the 1-D analogue of ``local_hydrostatic_residual`` in the current
    Fortran reconstruction.  The 2-D wrapper supplies the true four-neighbour
    residual when reproducing the production Cartesian algorithm.
    """
    h=np.asarray(h_center,dtype=float)
    B=np.asarray(B_center,dtype=float)
    if h.shape != B.shape or h.ndim != 1:
        raise ValueError('h_center and B_center must be matching 1-D arrays')
    eta=h+B
    r=np.zeros_like(h)
    eps=np.finfo(float).eps
    for i in range(h.size):
        num=0.0; den=0.0
        if i>0:
            num += abs(eta[i-1]-eta[i])
            den += abs(h[i-1]-h[i]) + abs(B[i-1]-B[i])
        if i+1<h.size:
            num += abs(eta[i+1]-eta[i])
            den += abs(h[i+1]-h[i]) + abs(B[i+1]-B[i])
        scale=128.0*eps*max(1.0,abs(eta[i]),abs(h[i]),abs(B[i]))
        r[i]=min(1.0,max(0.0,num/den)) if den>scale else 0.0
    return r

def topographic_relief_ratio_1d(h_center, B_minus, B_plus, dry_h_tol):
    h=np.asarray(h_center,dtype=float)
    Bm=np.asarray(B_minus,dtype=float); Bp=np.asarray(B_plus,dtype=float)
    relief=np.abs(Bp-Bm)
    return relief/np.maximum(h,float(dry_h_tol))

def raw_thickness(q, p):
    """Physical thickness from a possibly unsanitized conservative state."""
    q = np.asarray(q, dtype=float)
    return (q[..., 0] - q[..., 2]) / p.rho_c + q[..., 2] / p.rho_s

def transmissive_boundary(side, t, q_edge, p):
    return q_edge.copy()

def assemble_face_states(q, B, t, p, boundary, q_is_sanitized=False,
                         hydrostatic_residual=None,
                         topographic_relief_ratio=None):
    """Reconstruct the baseline HP-PCCU face states on a continuous bed.

    The direct-h and eta=h+B reconstructions are blended using the shared
    geometric cell endpoints from ``BedGeometry1D`` for the positivity-preserving
    eta slope.
    """
    q = np.asarray(q, dtype=float) if q_is_sanitized else sanitize(q, p)
    prim_q = primitive(q, p)
    (qL_cell, qR_cell, hL_cell, hR_cell, hL_raw, hR_raw,
     huL_cell, huR_cell, uL_cell, uR_cell) = reconstruct_state(
        q, p, return_physical=True, primitive_values=prim_q
    )
    B_cent = bed_center_values(B)
    if len(B_cent) != len(q):
        raise ValueError(f'bed geometry has {len(B_cent)} cells but q has {len(q)} states')
    BL_cell, BR_cell = reconstruct_bed(B, p)

    h_cent = prim_q[3]
    eta_cent = h_cent + B_cent
    if p.reconstruct_eta:
        etaL_cell, etaR_cell = reconstruct_array(eta_cent, p)
    else:
        etaL_cell = hL_cell + BL_cell
        etaR_cell = hR_cell + BR_cell

    if isinstance(B, BedGeometry1D):
        Bleft_endpoint = B.B_left_cell
        Bright_endpoint = B.B_right_cell
    else:
        # Convenience fallback for standalone cell-centred tests.  The Fortran
        # reference path should use BedGeometry1D/shared face geometry.
        Bleft_endpoint = BL_cell
        Bright_endpoint = BR_cell

    # Positivity-preserving eta candidate.
    eta_slope = etaR_cell - etaL_cell
    slope_min = 2.0 * (Bright_endpoint - eta_cent)
    slope_max = 2.0 * (eta_cent - Bleft_endpoint)
    slope_pp = np.minimum(np.maximum(eta_slope, slope_min), slope_max)
    etaL_pp = eta_cent - 0.5 * slope_pp
    etaR_pp = eta_cent + 0.5 * slope_pp
    hL_eta_raw = etaL_pp - Bleft_endpoint
    hR_eta_raw = etaR_pp - Bright_endpoint
    hL_eta = np.maximum(hL_eta_raw, 0.0)
    hR_eta = np.maximum(hR_eta_raw, 0.0)

    # Parameter-free direct-h / eta continuity blend.
    hL_orig = hL_cell.copy()
    hR_orig = hR_cell.copy()
    huL_orig = huL_cell.copy()
    huR_orig = huR_cell.copy()
    ncell = len(h_cent)
    Eh = np.zeros(ncell, dtype=float)
    Eeta = np.zeros(ncell, dtype=float)
    if ncell >= 2:
        jh = np.abs(hR_orig[:-1] - hL_orig[1:])
        je = np.abs(hR_eta[:-1] - hL_eta[1:])
        Eh[:-1] += jh; Eh[1:] += jh
        Eeta[:-1] += je; Eeta[1:] += je
    den = Eh + Eeta
    tol_dist = 1.0e-14*np.maximum(1.0, np.asarray(h_cent, dtype=float))
    w_eta = np.where(den > tol_dist, Eh/np.maximum(den, tol_dist), 0.5)

    # Residual-gated dry-safe selection, aligned with the current Fortran
    # HP reconstruction.  Dry cell averages always use eta.  A wet cell whose
    # eta candidate touches the bed uses eta only for hydrostatic-like states;
    # dynamic drainage retains the direct reconstruction.  Fully wet dynamic
    # cells suppress eta only when within-cell relief exceeds local thickness.
    if hydrostatic_residual is None:
        hydrostatic_residual = local_hydrostatic_residual_1d(h_cent, B_cent)
    else:
        hydrostatic_residual = np.asarray(hydrostatic_residual,dtype=float)
    if topographic_relief_ratio is None:
        topographic_relief_ratio = topographic_relief_ratio_1d(
            h_cent, Bleft_endpoint, Bright_endpoint, p.dry_h_tol)
    else:
        topographic_relief_ratio = np.asarray(topographic_relief_ratio,dtype=float)
    if hydrostatic_residual.shape != h_cent.shape:
        raise ValueError('hydrostatic_residual must match cell array')
    if topographic_relief_ratio.shape != h_cent.shape:
        raise ValueError('topographic_relief_ratio must match cell array')

    dry_center = h_cent <= p.dry_h_tol
    eta_touches_bed = (hL_eta <= p.dry_h_tol) | (hR_eta <= p.dry_h_tol)
    dynamic = hydrostatic_residual > p.hp_dynamic_residual_threshold
    w_eta = np.where(dry_center, 1.0, w_eta)
    w_eta = np.where((~dry_center) & eta_touches_bed,
                     np.where(dynamic,0.0,1.0), w_eta)
    rough_dynamic = ((~dry_center) & (~eta_touches_bed) & dynamic &
                     (topographic_relief_ratio > 1.0))
    w_eta = np.where(rough_dynamic,0.0,w_eta)

    hL_cell = (1.0-w_eta)*hL_orig + w_eta*hL_eta
    hR_cell = (1.0-w_eta)*hR_orig + w_eta*hR_eta

    # On locally flat geometry keep the original momentum reconstruction.
    huL_eta = hL_eta * uL_cell
    huR_eta = hR_eta * uR_cell
    geom_momentum = (np.abs(Bright_endpoint - Bleft_endpoint) > p.bed_step_tol).astype(float)
    w_hu = w_eta * geom_momentum
    huL_cell = (1.0-w_hu)*huL_orig + w_hu*huL_eta
    huR_cell = (1.0-w_hu)*huR_orig + w_hu*huR_eta
    etaL_cell = etaL_pp
    etaR_cell = etaR_pp
    hL_raw = hL_cell.copy()
    hR_raw = hR_cell.copy()

    yL, yR = reconstruct_array(prim_q[6], p)
    qL_cell = state_from_h_massfrac_hu(hL_cell, yL, huL_cell, p)
    qR_cell = state_from_h_massfrac_hu(hR_cell, yR, huR_cell, p)

    # Conservative momentum-slope admissibility.
    hi = prim_q[3]
    ui = prim_q[1]
    hui = hi * ui
    un = np.vstack((ui, np.r_[ui[0],ui[:-1]], np.r_[ui[1:],ui[-1]]))
    umin, umax = un.min(axis=0), un.max(axis=0)
    dh = 0.5*(hR_cell-hL_cell)
    dm_target = 0.5*(huR_cell-huL_cell)-ui*dh
    dm_lo = np.maximum((ui-umax)*hL_cell, (umin-ui)*hR_cell)
    dm_hi = np.minimum((ui-umin)*hL_cell, (umax-ui)*hR_cell)
    dm = np.minimum(np.maximum(dm_target,dm_lo),dm_hi)
    huL_cell = ui*hL_cell-dm
    huR_cell = ui*hR_cell+dm
    yL,yR = reconstruct_array(prim_q[6],p)
    qL_cell = state_from_h_massfrac_hu(hL_cell,yL,huL_cell,p)
    qR_cell = state_from_h_massfrac_hu(hR_cell,yR,huR_cell,p)
    uL_cell = primitive(qL_cell,p)[1]
    uR_cell = primitive(qR_cell,p)[1]

    q_left_ghost = boundary("left", t, q[0], p)
    q_right_ghost = boundary("right", t, q[-1], p)
    vals_left_ghost = primitive(q_left_ghost[None, :], p)
    vals_right_ghost = primitive(q_right_ghost[None, :], p)
    h_left_ghost = float(vals_left_ghost[3][0])
    h_right_ghost = float(vals_right_ghost[3][0])
    u_left_ghost = float(vals_left_ghost[1][0])
    u_right_ghost = float(vals_right_ghost[1][0])
    hu_left_ghost = h_left_ghost * u_left_ghost
    hu_right_ghost = h_right_ghost * u_right_ghost

    qL_ext = np.vstack([q_left_ghost, qL_cell, q_right_ghost])
    qR_ext = np.vstack([q_left_ghost, qR_cell, q_right_ghost])
    qL_face = qR_ext[:-1]
    qR_face = qL_ext[1:]
    uL_face = np.concatenate([[u_left_ghost], uR_cell])
    uR_face = np.concatenate([uL_cell, [u_right_ghost]])
    huL_face = np.concatenate([[hu_left_ghost], huR_cell])
    huR_face = np.concatenate([huL_cell, [hu_right_ghost]])

    if isinstance(B, BedGeometry1D):
        BgeomL_face = B.B_face.copy()
        BgeomR_face = B.B_face.copy()
    else:
        BgeomL_face = np.concatenate([[BL_cell[0]], BR_cell[:-1], [BR_cell[-1]]])
        BgeomR_face = np.concatenate([[BL_cell[0]], BL_cell[1:], [BR_cell[-1]]])

    # Boundary ghost eta retains the baseline zero-gradient FV convention.
    eta_left_ghost = h_left_ghost + BL_cell[0]
    eta_right_ghost = h_right_ghost + BR_cell[-1]
    etaL_face = np.concatenate([[eta_left_ghost], etaR_cell[:-1], [etaR_cell[-1]]])
    etaR_face = np.concatenate([[etaL_cell[0]], etaL_cell[1:], [eta_right_ghost]])

    primL_face = primitive(qL_face, p)
    primR_face = primitive(qR_face, p)
    BhydL_face = etaL_face - primL_face[3]
    BhydR_face = etaR_face - primR_face[3]

    return {
        "qL_cell": qL_cell, "qR_cell": qR_cell,
        "qL_face": qL_face, "qR_face": qR_face,
        "uL_cell": uL_cell, "uR_cell": uR_cell,
        "uL_face": uL_face, "uR_face": uR_face,
        "huL_cell": huL_cell, "huR_cell": huR_cell,
        "huL_face": huL_face, "huR_face": huR_face,
        "BL_cell": BL_cell, "BR_cell": BR_cell,
        "BgeomL_face": BgeomL_face, "BgeomR_face": BgeomR_face,
        "etaL_cell": etaL_cell, "etaR_cell": etaR_cell,
        "etaL_face": etaL_face, "etaR_face": etaR_face,
        "BhydL_face": BhydL_face, "BhydR_face": BhydR_face,
        "prim_q": prim_q,
        "primL_face": primL_face, "primR_face": primR_face,
        "momentum_slope_limited_cells": int(np.sum(dm != dm_target)),
        "hu_pair_mean_residual": float(np.max(np.abs(0.5*(huL_cell+huR_cell)-hui))),
        "min_momentum_interval_width": float(np.min(dm_hi-dm_lo)),
        "min_h_reconstructed_raw": float(min(np.min(hL_raw), np.min(hR_raw))),
        "w_eta": w_eta,
        "hydrostatic_residual": hydrostatic_residual,
        "topographic_relief_ratio": topographic_relief_ratio,
    }


def prepare_stage_face_cache(q, B, t, p, boundary, q_is_sanitized=False,
                             slope_geometry=None, hydrostatic_residual=None,
                             topographic_relief_ratio=None):
    """Build baseline HP-PCCU face data once for one RK stage."""
    p.checked()
    q = np.asarray(q, dtype=float) if q_is_sanitized else sanitize(q, p)
    data = assemble_face_states(q, B, t, p, boundary, q_is_sanitized=True,
                                hydrostatic_residual=hydrostatic_residual,
                                topographic_relief_ratio=topographic_relief_ratio)
    qL_face, qR_face = data['qL_face'], data['qR_face']

    need_ls_geometry = p.slope_correction or p.curvature_term
    if need_ls_geometry:
        if slope_geometry is None:
            raise ValueError('slope/curvature correction requires stage-local SlopeGeometry1D')
        G_fit_center=np.asarray(slope_geometry.G_center,dtype=float)
        G_fit_face=np.asarray(slope_geometry.G_face,dtype=float)
        if G_fit_center.size != len(q) or G_fit_face.size != len(q)+1:
            raise ValueError('slope geometry does not match 1-D state')
        if p.slope_correction:
            G_center=G_fit_center
            G_face=G_fit_face
        else:
            G_center=np.ones(len(q),dtype=float)
            G_face=np.ones(len(q)+1,dtype=float)
    else:
        G_center=np.ones(len(q),dtype=float)
        G_face=np.ones(len(q)+1,dtype=float)

    H0, fL, fR, a_minus, a_plus = cu_flux_from_states(
        qL_face, qR_face, p,
        uL_aux=data['uL_face'], uR_aux=data['uR_face'],
        primitive_L=data['primL_face'], primitive_R=data['primR_face'],
        grav_coeff=G_face
    )
    return dict(q=q, t=float(t), data=data, H0=H0, fL=fL, fR=fR,
                a_minus=a_minus, a_plus=a_plus,
                G_center=G_center, G_face=G_face,
                slope_geometry=slope_geometry)


def max_dt(q, B, dx, t, p, boundary, return_diagnostics=False,
           stage_cache=None):
    """Baseline HP-PCCU CFL limit from CU one-sided wave speeds."""
    cache = (prepare_stage_face_cache(
                 q, B, t, p, boundary,
                 slope_geometry=(build_slope_geometry_1d(B,dx)
                                 if (p.slope_correction or p.curvature_term) else None))
             if stage_cache is None else stage_cache)
    data = cache['data']
    a_minus, a_plus = cache['a_minus'], cache['a_plus']
    speed_max = max(float(np.max(np.maximum(np.abs(a_minus), np.abs(a_plus)))), 1.0e-12)
    dt_cfl = p.cfl * dx / speed_max
    if return_diagnostics:
        return dt_cfl, {
            'speed_max': speed_max,
            'max_abs_u_face': max(float(np.max(np.abs(data['uL_face']))),
                                  float(np.max(np.abs(data['uR_face'])))),
            'stage_cache_reused': stage_cache is not None,
        }
    return dt_cfl


def rhs_spatial(q, B, dx, t, p, boundary, return_diagnostics=False,
                stage_cache=None, include_curvature_source=True):
    """Semidiscrete baseline HP-PCCU operator."""
    p.checked()
    cache = (prepare_stage_face_cache(
                 q, B, t, p, boundary,
                 slope_geometry=(build_slope_geometry_1d(B,dx)
                                 if (p.slope_correction or p.curvature_term) else None))
             if stage_cache is None else stage_cache)
    data = cache['data']
    qL_cell, qR_cell = data['qL_cell'], data['qR_cell']
    qL_face, qR_face = data['qL_face'], data['qR_face']
    H = cache['H0'].copy()
    a_minus, a_plus = cache['a_minus'], cache['a_plus']

    if p.zero_transport_at_rest:
        huL, huR = data['huL_face'], data['huR_face']
        resting = ((huL == 0.0) & (huR == 0.0)) if p.rest_flux_tol == 0.0 else (
            (np.abs(huL) <= p.rest_flux_tol) & (np.abs(huR) <= p.rest_flux_tol))
        H[resting, 0] = 0.0
        H[resting, 2] = 0.0

    Bpsi = path_hydrostatic_source(
        qL_face, qR_face, data['etaL_face'], data['etaR_face'], p,
        primitive_L=data['primL_face'], primitive_R=data['primR_face'],
        grav_coeff=cache['G_face'])
    Bj = path_hydrostatic_source(
        qL_cell, qR_cell, data['etaL_cell'], data['etaR_cell'], p,
        grav_coeff=cache['G_face'][:-1], grav_coeff_right=cache['G_face'][1:])
    GL, GR = pccu_effective_face_values(H, Bpsi, a_minus, a_plus, p.speed_eps)

    R = -(GL[1:] - GR[:-1]) / dx
    R += Bj / dx

    curvature_accel = np.zeros(len(R), dtype=float)
    curvature_source_momentum = np.zeros(len(R), dtype=float)
    if p.curvature_term and include_curvature_source:
        sg = cache.get('slope_geometry')
        if sg is None or sg.Bx_center is None or sg.Bxx_center is None:
            raise ValueError('curvature_term requires LS slope/curvature geometry')
        M, u, _, _, _, _, _, _, _ = primitive(cache['q'], p)
        curvature_accel = np.asarray(sg.Bxx_center) * u*u
        curvature_source_momentum = -cache['G_center'] * M * curvature_accel * np.asarray(sg.Bx_center)
        R[:,1] += curvature_source_momentum

    if return_diagnostics:
        return R, dict(
            H=H, G_left=GL, G_right=GR, Bpsi=Bpsi, Bj=Bj,
            path_pair_error=GR-GL-Bpsi,
            a_minus=a_minus, a_plus=a_plus,
            qL_face=qL_face, qR_face=qR_face,
            uL_face=data['uL_face'], uR_face=data['uR_face'],
            huL_face=data['huL_face'], huR_face=data['huR_face'],
            BhydL_face=data['BhydL_face'], BhydR_face=data['BhydR_face'],
            BgeomL_face=data['BgeomL_face'], BgeomR_face=data['BgeomR_face'],
            grav_coeff_center=cache['G_center'], grav_coeff_face=cache['G_face'],
            curvature_accel=curvature_accel,
            curvature_source_momentum=curvature_source_momentum,
            etaL_face=data['etaL_face'], etaR_face=data['etaR_face'],
            min_h_reconstructed_raw=data['min_h_reconstructed_raw'],
            w_eta=data['w_eta'],
            stage_cache_reused=stage_cache is not None,
        )
    return R


def step_ssprk2(q, B, dx, t, dt_cap, p, boundary,
                 return_diagnostics=False, max_retries=12):
    """Stagewise CFL/positivity-safe SSPRK2 for the baseline HP-PCCU operator."""
    q = sanitize(q, p)
    slope_geom0 = (build_slope_geometry_1d(B,dx)
                   if (p.slope_correction or p.curvature_term) else None)
    cache0 = prepare_stage_face_cache(q, B, t, p, boundary,
                                      q_is_sanitized=True,
                                      slope_geometry=slope_geom0)
    dt0, d0 = max_dt(q, B, dx, t, p, boundary,
                     return_diagnostics=True, stage_cache=cache0)
    dt = min(float(dt_cap), 0.99 * float(dt0))
    retries = 0
    tol_h = 5.0e-13 * max(1.0, float(np.max(cache0['data']['prim_q'][3])))
    R0 = rhs_spatial(q, B, dx, t, p, boundary, stage_cache=cache0)

    while True:
        q1_raw = q + dt * R0
        min_h1 = float(np.min(raw_thickness(q1_raw, p)))
        q1 = sanitize(q1_raw, p)
        slope_geom1 = (build_slope_geometry_1d(B,dx)
                       if (p.slope_correction or p.curvature_term) else None)
        cache1 = prepare_stage_face_cache(q1, B, t + dt, p, boundary,
                                          q_is_sanitized=True,
                                          slope_geometry=slope_geom1)
        dt1, d1 = max_dt(q1, B, dx, t + dt, p, boundary,
                         return_diagnostics=True, stage_cache=cache1)
        stage_cfl_ok = dt <= float(dt1) * (1.0 + 2.0e-13)

        if stage_cfl_ok and min_h1 >= -tol_h:
            R1 = rhs_spatial(q1, B, dx, t + dt, p, boundary, stage_cache=cache1)
            q2_raw = 0.5 * q + 0.5 * (q1 + dt * R1)
            min_h2 = float(np.min(raw_thickness(q2_raw, p)))
            if min_h2 >= -tol_h:
                q2 = sanitize(q2_raw, p)
                diag = dict(
                    dt=dt, dt_cfl_stage0=float(dt0), dt_cfl_stage1=float(dt1),
                    retries=retries, min_raw_h_stage1=min_h1,
                    min_raw_h_stage2=min_h2,
                    min_raw_M_stage1=float(np.min(q1_raw[:,0])),
                    min_raw_M_stage2=float(np.min(q2_raw[:,0])),
                    max_abs_u_face=max(d0['max_abs_u_face'],d1['max_abs_u_face']),
                    speed_max=max(d0['speed_max'],d1['speed_max']))
                return (q2,diag) if return_diagnostics else q2

        retries += 1
        if retries > max_retries:
            raise RuntimeError('SSPRK2 stagewise positivity/CFL check failed to converge')
        dt_new = min(0.8 * dt, 0.95 * float(dt1))
        if min_h1 < -tol_h:
            dt_new = min(dt_new, 0.5 * dt)
        if not (dt_new > 1.0e-15 and dt_new < dt):
            dt_new = 0.5 * dt
        dt = dt_new

