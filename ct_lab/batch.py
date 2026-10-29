"""Sequential, resumable analysis of real NIfTI pairs, with equal-case summaries."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ct_lab.core import discover, fingerprint, load_pair, metadata, stats, LABELS, mask_analysis


def analyze(row, components=False):
    image, mask = load_pair(row)
    x = image.get_fdata(dtype=np.float32)
    result = {'id': row['id'], 'image': row['image'], 'mask': row['mask'], 'geometry': metadata(image),
              'image_sha256': fingerprint(row['image']), 'image_statistics': stats(x),
              'mask_geometry_matches': mask is not None, 'anomalies': [], 'class_statistics': {}}
    g = result['geometry']
    if not g['HU_verified']: result['anomalies'].append('HU scale unverified')
    if g['spatial_unit'] == 'unknown': result['anomalies'].append('Physical spatial units unknown')
    if g['qform_sform_disagree']: result['anomalies'].append('qform/sform conflict')
    if result['image_statistics']['finite'] != x.size: result['anomalies'].append('Nonfinite image')
    if mask is not None:
        y = np.asanyarray(mask.dataobj)
        result['mask_sha256'] = fingerprint(row['mask'])
        result['unique_mask_values'] = np.unique(y).tolist()
        for label in np.unique(y):
            selected = x[y == label]
            result['class_statistics'][str(int(label))] = dict(stats(selected), fraction=float(selected.size/x.size),
                                                               name=LABELS.get(int(label), 'UNKNOWN'))
        if set(np.unique(y))-set(LABELS): result['anomalies'].append('Unknown mask labels')
        if set(LABELS)-set(np.unique(y)): result['anomalies'].append('Missing class labels')
        if components: result['mask_anatomy'] = mask_analysis(y, image.affine, g['spatial_unit'])
    else:
        result['anomalies'].append('Missing expert mask')
    return result


def run(root, output, components=False):
    output = Path(output)
    # Do not permit reports to overwrite original datasets.
    if output.resolve() == Path(root).resolve() or Path(root).resolve() in output.resolve().parents:
        raise ValueError('Choose an output directory outside source data')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in discover(root):
        cache = output / (row['id']+'.json')
        signature = {k: [str(Path(row[k]).resolve()), Path(row[k]).stat().st_size, Path(row[k]).stat().st_mtime_ns]
                     for k in ('image', 'mask') if Path(row[k]).is_file()}
        try:
            previous = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
            if previous.get('source_signature') == signature and previous.get('components_requested') == components and 'error' not in previous:
                result = previous
            else:
                result = analyze(row, components)
                result.update(source_signature=signature, components_requested=components)
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        except Exception as exc:
            result = {'id': row['id'], 'error': str(exc), 'anomalies': ['Read/integrity/geometry failure']}
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        records.append(result)
        print(row['id'], result.get('error', result.get('image_statistics', {}).get('mean')), flush=True)
    successful = [r for r in records if 'error' not in r]
    means = np.array([r['image_statistics']['mean'] for r in successful])
    if len(means):
        q1, q3 = np.percentile(means, [25, 75])
        for r, mean in zip(successful, means):
            if mean < q1-1.5*(q3-q1) or mean > q3+1.5*(q3-q1): r['anomalies'].append('Outlier: patient mean intensity (1.5 IQR)')
    summary = {'discovered_pairs': len(records), 'read_successful': len(successful), 'failed': len(records)-len(successful),
               'aggregation': 'Equal weight per released case; no pooled voxel training fit',
               'case_mean_distribution': stats(means), 'records': records}
    (output/'dataset.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'patients.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'mean', 'std', 'min', 'max', 'shape', 'unit', 'anomalies', 'error'])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(id=r['id'], **{k: r.get('image_statistics', {}).get(k) for k in ('mean','std','min','max')},
                                 shape=r.get('geometry', {}).get('shape_xyz'), unit=r.get('geometry', {}).get('spatial_unit'),
                                 anomalies='; '.join(r['anomalies']), error=r.get('error')))
    with (output/'classes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','label','name','count','fraction','mean','std','median','min','max'])
        writer.writeheader()
        for r in successful:
            for label, s in r['class_statistics'].items():
                writer.writerow(dict(id=r['id'], label=label, **{k:s.get(k) for k in writer.fieldnames[2:]}))
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[r['id'] for r in successful], y=means, mode='markers', name='One point per case'))
    fig.update_layout(title='Case mean source intensity (not verified HU)', xaxis_title='Released case', yaxis_title='Source intensity')
    html = fig.to_html(full_html=False, include_plotlyjs=True)
    import html as html_module
    (output/'report.html').write_text('<meta charset="utf-8"><h1>ImageCHD68: measured data</h1><p>HU and physical scale unverified.</p>'+html+
                                     '<pre>'+html_module.escape(json.dumps({k:v for k,v in summary.items() if k!='records'}, indent=2))+'</pre>', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--components', action='store_true', help='Expensive full-volume 2D/3D component analysis')
    args = parser.parse_args()
    run(args.data, args.output, args.components)
