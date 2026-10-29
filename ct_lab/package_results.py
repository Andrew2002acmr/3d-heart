"""Package measured results and reproducibility instructions, excluding raw CT/weights."""
from pathlib import Path
import zipfile


def main():
    repo=Path(__file__).resolve().parents[1]
    results=repo/'outputs/ct_preprocessing_lab'
    destination=repo/'outputs/CT_PREPROCESSING_LAB_RESULTS.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results.rglob('*')):
            if path.is_file(): archive.write(path,Path('results')/path.relative_to(results))
        for name in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','requirements-ct-lab.txt'):
            archive.write(repo/name,name)
        for path in sorted((repo/'ct_lab').glob('*')):
            if path.is_file() and path.suffix in ('.py','.md'): archive.write(path,Path('ct_lab')/path.name)
        archive.write(repo/'tests/test_ct_lab.py','tests/test_ct_lab.py')
    with zipfile.ZipFile(destination) as archive:
        failure=archive.testzip()
        if failure:raise ValueError('ZIP CRC failure: '+failure)
        print(destination, len(archive.namelist()), destination.stat().st_size,flush=True)


if __name__=='__main__':main()
