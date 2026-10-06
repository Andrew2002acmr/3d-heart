"""All axial positions, five physical planes; restore probabilities before thresholding."""
import json
from pathlib import Path
import numpy as np
import torch
from heart3d.ml.geometry import physical_context


@torch.inference_mode()
def predict_patient(model, directory, offsets_mm, batch_size, device):
    directory = Path(directory)
    provenance = json.loads((directory/'provenance.json').read_text())
    image = np.load(directory/'image.npy', mmap_mode='r')
    positions = provenance['transform']['z_positions_mm']
    if len(positions) != len(image):
        raise ValueError('Cached grid and provenance differ')
    model.eval()
    probability = np.empty(image.shape, dtype=np.float32)
    try:
        for begin in range(0, len(image), batch_size):
            contexts = []
            for center in range(begin, min(begin+batch_size, len(image))):
                lo, hi, weight, _ = physical_context(positions, center, offsets_mm)
                contexts.append(np.stack([(1-w)*image[a]+w*image[b] for a,b,w in zip(lo,hi,weight)]))
            tensor = torch.from_numpy(np.stack(contexts).astype(np.float32)).to(device)
            predicted = model(tensor).sigmoid()[:,0].cpu().numpy()
            if not np.isfinite(predicted).all():
                raise ValueError('Non-finite prediction')
            probability[begin:begin+len(predicted)] = predicted
    finally:
        image._mmap.close()
    return probability, provenance
