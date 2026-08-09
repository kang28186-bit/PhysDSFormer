import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PairedImageDataset(Dataset):
    """Load paired degraded/reference images from `cond` and `gt` directories."""

    def __init__(self, root, split="train", image_size=256, augment=False):
        self.root = Path(root) / split
        self.source_dir = self.root / "cond"
        self.target_dir = self.root / "gt"
        self.image_size = int(image_size)
        self.augment = bool(augment)

        if not self.source_dir.is_dir() or not self.target_dir.is_dir():
            raise FileNotFoundError(
                "Expected paired directories: {} and {}".format(
                    self.source_dir, self.target_dir
                )
            )

        source_files = {
            path.name: path
            for path in self.source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        target_files = {
            path.name: path
            for path in self.target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        missing_targets = sorted(set(source_files) - set(target_files))
        missing_sources = sorted(set(target_files) - set(source_files))
        if missing_targets or missing_sources:
            raise RuntimeError(
                "Unpaired files detected: {} source-only and {} target-only. "
                "Examples: source-only={}, target-only={}".format(
                    len(missing_targets),
                    len(missing_sources),
                    missing_targets[:3],
                    missing_sources[:3],
                )
            )

        names = sorted(source_files)
        if not names:
            raise RuntimeError("No paired images with matching file names were found")
        self.samples = [(source_files[name], target_files[name], name) for name in names]

    def __len__(self):
        return len(self.samples)

    def _transform_pair(self, source, target):
        size = (self.image_size, self.image_size)
        source = source.resize(size, Image.BICUBIC)
        target = target.resize(size, Image.BICUBIC)
        if self.augment:
            if random.random() < 0.5:
                source = ImageOps.mirror(source)
                target = ImageOps.mirror(target)
            rotations = random.randint(0, 3)
            if rotations:
                angle = 90 * rotations
                source = source.rotate(angle)
                target = target.rotate(angle)
        return source, target

    @staticmethod
    def _to_tensor(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor * 2.0 - 1.0

    def __getitem__(self, index):
        source_path, target_path, name = self.samples[index]
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")
        source, target = self._transform_pair(source, target)
        return {
            "source": self._to_tensor(source),
            "target": self._to_tensor(target),
            "filename": name,
        }
