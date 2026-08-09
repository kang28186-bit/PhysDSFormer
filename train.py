import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets import PairedImageDataset
from losses import MultiColorSpaceStructuralLoss
from models import MODEL_REGISTRY, build_model
from utils import (
    AverageMeter,
    batch_psnr,
    batch_ssim,
    load_model_state,
    save_training_checkpoint,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysDSFormer")
    parser.add_argument("--data-root", default="data", help="Parent dataset directory")
    parser.add_argument("--dataset", default="UIEB", choices=("UIEB", "LSUI", "EUVP"))
    parser.add_argument(
        "--model", default="physdsformer-tiny", choices=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--output", default="checkpoints", help="Checkpoint root")
    parser.add_argument("--resume", default=None, help="Resume from a training checkpoint")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-split", default=None, help="Override config")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config")
    parser.add_argument("--image-size", type=int, default=None, help="Override config")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_config(dataset):
    path = Path(__file__).resolve().parent / "configs" / "{}.json".format(dataset.lower())
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_one_epoch(loader, model, criterion, optimizer, scaler, device, use_amp):
    model.train()
    losses = AverageMeter()
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            prediction = model(source)
        with autocast(enabled=False):
            loss = criterion(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), source.shape[0])
    return losses.average


@torch.no_grad()
def validate(loader, model, device):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(source).clamp(-1.0, 1.0) * 0.5 + 0.5
        target = target * 0.5 + 0.5
        psnr_meter.update(batch_psnr(prediction, target).item(), source.shape[0])
        ssim_meter.update(batch_ssim(prediction, target).item(), source.shape[0])
    return psnr_meter.average, ssim_meter.average


def main():
    args = parse_args()
    config = load_config(args.dataset)
    if config.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only the AdamW optimizer is supported")
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    batch_size = args.batch_size or config["batch_size"]
    image_size = args.image_size or config["image_size"]
    validation_split = args.validation_split or config.get("validation_split", "val")
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    seed_everything(args.seed)

    dataset_root = Path(args.data_root) / args.dataset
    train_data = PairedImageDataset(
        dataset_root, split="train", image_size=image_size, augment=True
    )
    valid_data = PairedImageDataset(
        dataset_root, split=validation_split, image_size=image_size, augment=False
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.model).to(device)
    criterion = MultiColorSpaceStructuralLoss(**config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    scaler = GradScaler(enabled=use_amp)
    best_psnr = float("-inf")
    start_epoch = 1
    checkpoint_path = Path(args.output) / args.dataset.lower() / "{}.pth".format(args.model)
    last_checkpoint_path = (
        Path(args.output) / args.dataset.lower() / "{}-last.pth".format(args.model)
    )

    if args.resume:
        checkpoint = load_model_state(model, args.resume, device)
        if not isinstance(checkpoint, dict) or "optimizer" not in checkpoint:
            raise ValueError("Resume requires a complete training checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        if start_epoch > epochs:
            raise ValueError("Resume checkpoint has already reached the requested epochs")
        print("resumed={} next_epoch={}".format(args.resume, start_epoch))

    print("device={} train={} valid={}".format(device, len(train_data), len(valid_data)))
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, use_amp
        )
        scheduler.step()
        if epoch % config["eval_frequency"] != 0:
            print("epoch={}/{} loss={:.6f}".format(epoch, epochs, train_loss))
            save_training_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_psnr,
            )
            continue

        psnr, ssim = validate(valid_loader, model, device)
        print(
            "epoch={}/{} loss={:.6f} psnr={:.4f} ssim={:.5f}".format(
                epoch, epochs, train_loss, psnr, ssim
            )
        )
        if psnr > best_psnr:
            best_psnr = psnr
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_psnr,
            )
            print("saved={}".format(checkpoint_path))
        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_psnr,
        )


if __name__ == "__main__":
    main()
