"""Loopback-only independent NiiVue viewer of prepared original-grid CT/Heart."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--port', type=int, default=8766)
    p.add_argument('--reviews',type=Path,help='Optional reviewed status manifest')
    p.add_argument('--predictions',type=Path,help='Optional per-patient original-grid predictions directory')
    a=p.parse_args(); allowed={'/': Path(__file__).with_name('pediatric_ct_qa_viewer.html')}; cases=[]
    reviews={r['patient_id']:r for r in json.loads(a.reviews.read_text())['records']} if a.reviews else {}
    for path in sorted(a.data.glob('*/pilot_qa.json')):
        info=json.loads(path.read_text())
        if not info.get('mask_rasterized'): continue
        prepared=path.parent/'prepared'
        prediction=a.predictions/info['patient_id']/'heart_prediction_original.nii.gz' if a.predictions else None
        if prediction is not None and not prediction.is_file():continue
        if prediction is not None:
            gt=nib.load(prepared/'heart_gt_original.nii.gz');pred=nib.load(prediction)
            if gt.shape!=pred.shape or not np.allclose(gt.affine,pred.affine,atol=1e-5):
                raise ValueError(f'Prediction geometry mismatch: {info["patient_id"]}')
            allowed[f'/data/{info["patient_id"]}/heart_prediction_original.nii.gz']=prediction
        # Crosshair position is NiiVue's canonical RAS fraction, not storage indices.
        image=nib.as_closest_canonical(nib.load(prepared/'heart_gt_original.nii.gz'))
        voxels=np.argwhere(np.asarray(image.dataobj)>0)
        center=(np.median(voxels, axis=0)+.5)/image.shape
        cases.append({k:info[k] for k in ('patient_id','age','scanner','status')})
        if info['patient_id'] in reviews: cases[-1]['status']=reviews[info['patient_id']]['status']
        cases[-1]['center_fraction']=center.tolist()
        cases[-1]['prediction_available']=prediction is not None
        cases[-1]['physical_geometry_confirmed']=bool(info.get('CT',{}).get('physical_geometry_confirmed_from_DICOM'))
        for name in ('ct_original.nii.gz','heart_gt_original.nii.gz'):
            allowed[f'/data/{info["patient_id"]}/{name}']=prepared/name
    payload=json.dumps(cases).encode()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url=self.path.split('?')[0]
            if url=='/cases.json':
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
            path=allowed.get(url)
            if path is None or not path.is_file(): self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if path.suffix=='.html' else 'application/octet-stream')
            self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
            with path.open('rb') as stream: shutil.copyfileobj(stream,self.wfile)
        def log_message(self,*args): pass
    print(f'http://127.0.0.1:{a.port} | {len(cases)} prepared cases',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__': main()
