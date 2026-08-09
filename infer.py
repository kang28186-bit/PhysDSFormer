import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import MODEL_REGISTRY, build_model
from utils import load_model_state, save_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance underwater images")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth checkpoint")
    parser.add_argument(
        "--model", default="physdsformer-tiny", choices=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def image_to_tensor(path, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def collect_images(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image extension: {}".format(path.suffix))
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.model).to(device).eval()
    load_model_state(model, args.checkpoint, device)

    output_dir = Path(args.output)
    images = collect_images(args.input)
    if not images:
        raise RuntimeError("No images were found")

    with torch.no_grad():
        for image_path in images:
            prediction = model(image_to_tensor(image_path, device)).clamp(-1.0, 1.0)
            prediction = prediction[0] * 0.5 + 0.5
            save_image(prediction, output_dir / image_path.name)
            print("Saved {}".format(output_dir / image_path.name))


if __name__ == "__main__":
    main()
