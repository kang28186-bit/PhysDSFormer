import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedImageDataset
from models import MODEL_REGISTRY, build_model
from utils import AverageMeter, batch_psnr, batch_ssim, load_model_state, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PhysDSFormer")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="UIEB", choices=("UIEB", "LSUI", "EUVP"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model", default="physdsformer-tiny", choices=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--output", default="results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.model).to(device).eval()
    load_model_state(model, args.checkpoint, device)
    dataset = PairedImageDataset(
        Path(args.data_root) / args.dataset,
        split="test",
        image_size=args.image_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_root = Path(args.output) / args.dataset / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    rows = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True) * 0.5 + 0.5
        prediction = model(source).clamp(-1.0, 1.0) * 0.5 + 0.5
        psnr = batch_psnr(prediction, target).item()
        ssim = batch_ssim(prediction, target).item()
        psnr_meter.update(psnr)
        ssim_meter.update(ssim)
        filename = batch["filename"][0]
        rows.append((filename, psnr, ssim))
        if args.save_images:
            save_image(prediction[0], output_root / "images" / filename)

    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "psnr", "ssim"))
        writer.writerows(rows)
        writer.writerow(("AVERAGE", psnr_meter.average, ssim_meter.average))
    print("PSNR={:.4f} SSIM={:.5f}".format(psnr_meter.average, ssim_meter.average))
    print("metrics={}".format(metrics_path))


if __name__ == "__main__":
    main()

