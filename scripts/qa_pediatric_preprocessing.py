"""Reproducible train-only physical-context mosaic; not prediction evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from heart3d.ml.dataset import HeartDataset
from heart3d.pediatric import write_json
from heart3d.storage import require_space


def render(config, data, output):
    dataset = HeartDataset(config, 'train', data, augmentation=False)
    chosen = {}
    for row in dataset.rows:
        chosen.setdefault(row['age_group'], row['patient_id'])
    require_space(dataset.root, 10_000_000, int(dataset.config['reserve_GB'] * 1e9))
    figure, axes = plt.subplots(len(chosen), 6, figsize=(18, 3 * len(chosen)))
    low, high = dataset.preproc['HU_clip']
    window = [(value-low) * 2/(high-low)-1 for value in (-150, 250)]
    records = []
    for row_index, (group, patient) in enumerate(sorted(chosen.items())):
        positive = dataset.provenance[patient]['positive_indices']
        index = dataset.indices.index((patient, positive[len(positive)//2]))
        sample = dataset[index]
        for column, plane in enumerate(sample['image']):
            axes[row_index, column].imshow(plane, cmap='gray', vmin=window[0], vmax=window[1])
            axes[row_index, column].set_title(f"{group} | {dataset.preproc['context_offsets_mm'][column]:+g} mm")
        axes[row_index, 5].imshow(sample['image'][2], cmap='gray', vmin=window[0], vmax=window[1])
        axes[row_index, 5].contour(sample['target'][0], levels=[.5], colors=['lime'], linewidths=.8)
        axes[row_index, 5].set_title(patient + '\ncentral source GT')
        for axis in axes[row_index]:
            axis.axis('off')
        records.append({'patient_id': patient, 'age_group': group,
            'center_index': sample['center_index'], 'center_position_mm': sample['center_position_mm'],
            'context_outside_scan': sample['context_outside_scan'].tolist()})
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(output, dpi=120); plt.close(figure)
    write_json(output.with_suffix('.json'), {'partition': 'train', 'augmentation': False, 'records': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); render(args.config, args.data, args.out)
