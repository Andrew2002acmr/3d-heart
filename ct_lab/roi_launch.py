import ct_lab.roi_core as core
import runpy
if __name__=='__main__':
    core.seed_runtime(core.SEED,1,'cpu');runpy.run_module('ct_lab.launch',run_name='__main__')
