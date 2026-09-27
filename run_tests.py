from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
files=[
 ROOT/'tests'/'test_1d_drysafe.py',
 ROOT/'tests'/'test_1d_mass_fraction.py',
 ROOT/'tests'/'test_2d_mass_fraction.py',
 ROOT/'tests'/'test_hp_pccu_2d_core.py',
 ROOT/'tests'/'test_hp_pccu_2d_imex_ap.py',
]
for f in files:
 print('RUN',f.relative_to(ROOT));runpy.run_path(str(f),run_name='__main__')
print('ALL TESTS PASSED')
