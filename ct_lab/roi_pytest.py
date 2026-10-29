"""Keep legacy regression JIT caches out of frozen experiment outputs."""
import os
from pathlib import Path


def pytest_collection_finish(session):
    os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/heart_localization_roi_05/numba_cache')
