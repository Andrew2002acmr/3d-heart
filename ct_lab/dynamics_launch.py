import runpy
from ct_lab.dynamics_core import seed_runtime,OUT,os
if __name__=='__main__':
    seed_runtime(20261009,1,'cpu');os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache');runpy.run_module('ct_lab.launch',run_name='__main__')
