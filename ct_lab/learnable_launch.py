"""Existing CT Lab with isolated stage-03 runtime/cache, no automatic training."""
import ct_lab.learnable_core
import runpy
if __name__=='__main__':runpy.run_module('ct_lab.launch',run_name='__main__')
