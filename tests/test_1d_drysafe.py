import numpy as np
import hp_pccu_1d_core as hp
p=hp.SolverParams(dry_h_tol=1e-10,hp_dynamic_residual_threshold=0.90).checked()
B=hp.BedGeometry1D(np.array([0.0,0.2,0.45,1.35,1.35,0.85,0.65,0.55]));H0=3.0;h=H0-B.B_center
q=hp.state_from_h_alpha_hu(h,0.0,np.zeros_like(h),p)
d=hp.assemble_face_states(q,B,0.0,p,hp.transmissive_boundary,hydrostatic_residual=np.zeros_like(h),topographic_relief_ratio=np.zeros_like(h))
assert np.max(np.abs(d['etaL_cell']-H0))<2e-13
assert np.max(np.abs(d['etaR_cell']-H0))<2e-13
B=hp.BedGeometry1D(np.array([0.0,0.0,2.0,2.0]));h=np.full(3,0.2);q=hp.state_from_h_alpha_hu(h,0.0,np.zeros_like(h),p)
d=hp.assemble_face_states(q,B,0.0,p,hp.transmissive_boundary,hydrostatic_residual=np.ones(3),topographic_relief_ratio=np.full(3,10.0))
assert d['w_eta'][1]==0.0
assert abs(hp.primitive(d['qR_cell'],p)[3][1]-0.2)<5e-14
print('PASS: 1-D dry-safe HP reconstruction')
