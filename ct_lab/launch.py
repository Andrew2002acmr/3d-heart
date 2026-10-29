"""Use existing scientific runtime plus optional workspace-local UI dependencies."""
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEPENDENCIES = REPO.parent / 'ct_preprocessing_deps'
if DEPENDENCIES.exists():
    sys.path.append(str(DEPENDENCIES))
FEATURE_DEPENDENCIES = REPO.parent / 'ct_feature_deps'
sys.path.append(str(FEATURE_DEPENDENCIES))
os.environ.setdefault('NUMBA_CACHE_DIR',str(REPO/'outputs/filtering_feature_01/numba_cache'))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        import runpy
        sys.argv.pop(1)
        runpy.run_module('ct_lab.batch', run_name='__main__')
    else:
        from streamlit.web import cli
        sys.argv = ['streamlit', 'run', str(REPO/'ct_lab/app.py'), '--server.address=127.0.0.1',
                    '--server.port=8510', '--server.headless=true', '--global.developmentMode=false',
                    '--browser.gatherUsageStats=false', *sys.argv[1:]]
        cli.main()
