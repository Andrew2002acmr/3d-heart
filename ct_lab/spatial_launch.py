"""Run the existing lab with stage-02 cache isolation."""
import os
from pathlib import Path
import runpy
os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/spatial_feature_02/numba_cache')
if __name__=='__main__':runpy.run_module('ct_lab.launch',run_name='__main__')
