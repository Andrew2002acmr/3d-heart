"""Launch the existing lab with isolated stage04 caches and one CPU compute thread."""
import ct_lab.multiclass_core as core
import runpy
if __name__=='__main__':
    core.seed_all(core.SEEDS[0]);runpy.run_module('ct_lab.launch',run_name='__main__')
