import os
from pathlib import Path
def pytest_collection_finish(session):
    os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/training_dynamics_06/numba_cache')
def pytest_runtest_setup(item):
    os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/training_dynamics_06/numba_cache')
def pytest_runtest_teardown(item,nextitem):
    os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/training_dynamics_06/numba_cache')
