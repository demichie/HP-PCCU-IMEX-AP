import numpy as np
import hp_pccu_1d_core as hp1
import hp_pccu_2d_mass_fraction_core as hp2
p=hp1.SolverParams().checked()
q=hp2.state2d_from_h_massfrac_uv(np.array([1.,2.]),np.array([0.2,0.6]),0.3,-0.1,p)
pr=hp2.primitive2d(q,p)
assert np.allclose(pr['y_s'],[0.2,0.6],rtol=2e-14,atol=2e-14)
print('PASS: 2-D mass-fraction state conversion')
