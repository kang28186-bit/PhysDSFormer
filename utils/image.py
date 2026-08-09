from pathlib import Path

import numpy as np
from PIL import Image


def save_image(tensor, path):
    """Save a CHW tensor in [0, 1] as an RGB image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.round(array.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)

