"""Post-hoc original-grid multiclass CT/GT/prediction QA; never tunes the model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from heart3d.labels import LABELS
from heart3d.ml.cardiac_data import read_json, write_json
from heart3d.storage import sha256_file, require_space


def overlay(labels):
    result = np.zeros((*labels.shape, 4), dtype=np.float32)
    for label, (_, _, color) in LABELS.items():
        result[labels == label] = to_rgba(color, .55)
    result[labels > 7] = to_rgba('#a0a0a0', .3)
    return result


def errors(gt, pred):
    result = np.zeros((*gt.shape, 4), dtype=np.float32)
    valid = gt <= 7
    foreground = (gt > 0) & valid
    result[foreground & (pred == 0)] = to_rgba('#ff4040', .7)
    result[valid & (gt == 0) & (pred > 0)] = to_rgba('#40b7ff', .7)
    result[foreground & (pred > 0) & (pred != gt)] = to_rgba('#ffdf40', .7)
    return result


def draw_row(axes, ct, gt, pred, axis, index, clip, title):
    planes = [np.take(v, index, axis=axis).T for v in (ct, gt, pred)]
    for column, ax in enumerate(axes):
        ax.imshow(planes[0], cmap='gray', vmin=clip[0], vmax=clip[1], origin='lower', interpolation='nearest')
        if column in (1, 2):
            ax.imshow(overlay(planes[column]), origin='lower', interpolation='nearest')
        if column == 3:
            ax.imshow(errors(planes[1], planes[2]), origin='lower', interpolation='nearest')
        ax.set_title(title + '\n' + ['КТ', 'Исходная разметка (GT)', 'Предсказание', 'Ошибки относительно GT'][column], fontsize=10)
        ax.axis('off')


def decorate(fig, title, include_errors=True):
    handles = [Patch(facecolor=color, label=short) for short, _, color in LABELS.values()]
    if include_errors:
        handles += [Patch(facecolor=c, label=t) for c, t in [('#ff4040', 'Пропуск'), ('#40b7ff', 'Лишняя область'), ('#ffdf40', 'Перепутана структура')]]
    fig.legend(handles=handles, loc='lower center', ncol=7 if not include_errors else 5, fontsize=10, frameon=False, bbox_to_anchor=(.5, .002))
    fig.suptitle(title + '\nOriginal NIfTI index grid; физический масштаб не подтверждён', fontsize=13)


def render_case(row, metric, role, data, evaluation, output, clip, checkpoint_sha):
    case = row['case_id']
    sources = [data / row['image_relative_path'], data / row['mask_relative_path'], evaluation / case / 'cardiac_prediction_original.nii.gz']
    expected = [row['image_SHA256'], row['mask_SHA256'], read_json(evaluation / case / 'provenance.json')['prediction_SHA256']]
    for path, digest in zip(sources, expected):
        if sha256_file(path) != digest:
            raise ValueError('Source/GT/prediction SHA mismatch')
    provenance = read_json(evaluation / case / 'provenance.json')
    if provenance['checkpoint_SHA256'] != checkpoint_sha:
        raise ValueError('Prediction checkpoint differs')
    loaded = [nib.load(path) for path in sources]
    if any(v.shape != loaded[0].shape or not np.allclose(v.affine, loaded[0].affine, atol=1e-5, rtol=0) for v in loaded[1:]):
        raise ValueError('Original CT/GT/prediction grid differs')
    ct = loaded[0].get_fdata(dtype=np.float32)
    gt = np.asanyarray(loaded[1].dataobj)
    pred = np.asanyarray(loaded[2].dataobj)
    if not np.isfinite(ct).all() or np.any((pred < 0) | (pred > 7)):
        raise ValueError('Invalid data or prediction labels')
    foreground = (gt > 0) & (gt <= 7)
    positive = [np.flatnonzero(foreground.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(not len(p) for p in positive):
        raise ValueError('Empty cardiac reference')
    center = [int((p[0] + p[-1]) // 2) for p in positive]
    del foreground
    specs = [('Axial 20%', 2, int(positive[2][round(.2 * (len(positive[2]) - 1))])),
             ('Axial 50%', 2, int(positive[2][round(.5 * (len(positive[2]) - 1))])),
             ('Axial 80%', 2, int(positive[2][round(.8 * (len(positive[2]) - 1))])),
             ('Coronal', 1, center[1]), ('Sagittal', 0, center[0])]
    title = f'{role}: {case} | case macro Dice={metric["macro_foreground_Dice"]:.3f}'
    fig, axes = plt.subplots(5, 4, figsize=(15, 18))
    for axes_row, (name, axis, index) in zip(axes, specs):
        draw_row(axes_row, ct, gt, pred, axis, index, clip, f'{name}, index {index}')
    decorate(fig, title)
    fig.tight_layout(rect=(0, .055, 1, .955))
    fig.savefig(output / f'{case}_comparison.png', dpi=115)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    draw_row(axes, ct, gt, pred, 2, specs[1][2], clip, f'Axial index {specs[1][2]}')
    decorate(fig, title, include_errors=False)
    fig.tight_layout(rect=(0, .06, 1, .9))
    fig.savefig(output / f'{case}_preview.jpg', dpi=110, pil_kwargs={'quality': 90})
    plt.close(fig)
    pa_missed = ((gt == 7) & (pred != 7)).sum(axis=(0, 1))
    pa_index = int(np.argmax(pa_missed))
    if pa_missed[pa_index] > 0:
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        draw_row(axes, ct, gt, pred, 2, pa_index, clip, f'Axial index {pa_index}')
        decorate(fig, title + ' | максимум пропусков PA на axial slice')
        fig.tight_layout(rect=(0, .09, 1, .9))
        fig.savefig(output / f'{case}_PA_failure.png', dpi=110)
        plt.close(fig)
    result = {'case_id': case, 'selection_role': role, 'metric': metric,
              'case_selection': 'post-hoc best/upper-median/worst macro foreground test Dice; descriptive examples, no model tuning',
              'slice_sampling': specs, 'PA_failure_slice': pa_index,
              'PA_failure_slice_selection': 'maximum count of GT PA voxels not predicted as PA; post-hoc error visualization',
              'CT_display_clip': list(clip), 'display_clip_source': 'unchanged train-only preprocessing range; not HU',
              'physical_geometry_verified': False, 'image_geometry': 'original array index aspect; no mm scale bar',
              'source_files': [{'path': str(p), 'SHA256': d} for p, d in zip(sources, expected)],
              'checkpoint_SHA256': checkpoint_sha, 'source_data_modified': False, 'postprocessing_applied': False,
              'ignored_gt_labels': 'labels >7 have gray overlay and are excluded from error highlighting'}
    write_json(output / f'{case}_QA.json', result)
    print('RENDERED', case, role, flush=True)
    return result


def charts(metrics, history, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['training_loss'] for r in history], color='#1660ad')
    axes[0].set(xlabel='Epoch', ylabel='Train CE+Dice loss', title='Training loss')
    axes[1].plot(epochs, [r['validation_mean_case_foreground_Dice'] for r in history], color='#30936b')
    axes[1].set(xlabel='Epoch', ylabel='Validation mean-case macro Dice', title='Validation (prepared grid)', ylim=(0, 1))
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classes = list(metrics['per_class_mean_Dice'])
    bars = axes[0].bar(classes, [metrics['per_class_mean_Dice'][c] for c in classes], color=[LABELS[i + 1][2] for i in range(7)])
    axes[0].bar_label(bars, fmt='%.3f', fontsize=9)
    axes[0].set(title='Test: mean-case Dice by structure', ylim=(0, 1))
    records = sorted(metrics['records'], key=lambda r: r['macro_foreground_Dice'])
    axes[1].barh([r['case_id'] for r in records], [r['macro_foreground_Dice'] for r in records], color='#548eae')
    axes[1].set(title='Test: case macro foreground Dice', xlim=(0, 1))
    fig.suptitle('CHD68 v1 | original release grid | 10 test cases | no age/physical-scale claim')
    fig.tight_layout(); fig.savefig(output / 'test_metrics.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'evaluation', 'training', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cohort', type=Path, default=Path('metadata/pediatric/cardiac_chd68_cohort_v1.json'))
    parser.add_argument('--preprocessing', type=Path, default=Path('configs/cardiac_chd68_preprocessing_v1.json'))
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('QA output exists; choose a new directory')
    require_space(args.output, 100_000_000, 80_000_000_000)
    metrics = read_json(args.evaluation / 'metrics.json')
    history = read_json(args.training / 'history.json')
    cohort = {r['case_id']: r for r in read_json(args.cohort)['records']}
    preproc = read_json(args.preprocessing)
    ordered = sorted(metrics['records'], key=lambda r: (r['macro_foreground_Dice'], r['case_id']))
    if metrics['partition'] != 'test' or not ordered or any(r['case_id'] not in cohort for r in ordered):
        raise ValueError('Expected completed test metrics from this cohort')
    clip = np.asarray(preproc['clip_source_intensity'], dtype=float)
    if clip.shape != (2,) or not np.isfinite(clip).all() or clip[0] >= clip[1]:
        raise ValueError('Invalid train-only display range')
    if preproc['fitted_partition'] != 'train' or metrics['preprocessing_SHA256'] != sha256_file(args.preprocessing) or metrics['cohort_SHA256'] != sha256_file(args.cohort):
        raise ValueError('Frozen preprocessing/cohort differs from evaluated protocol')
    args.output.mkdir(parents=True)
    selected = [('Лучший test-случай', ordered[-1]), ('Средний по рангу test-случай', ordered[len(ordered) // 2]), ('Самый слабый test-случай', ordered[0])]
    results = [render_case(cohort[r['case_id']], r, role, args.data, args.evaluation, args.output, preproc['clip_source_intensity'], metrics['checkpoint_SHA256']) for role, r in selected]
    charts(metrics, history, args.output)
    write_json(args.output / 'QA_summary.json', {'source_metrics_SHA256': sha256_file(args.evaluation / 'metrics.json'), 'cases': results})


if __name__ == '__main__': main()
