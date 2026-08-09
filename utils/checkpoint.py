from pathlib import Path

import torch


def normalize_state_dict(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model_state(model, path, device, strict=True):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(normalize_state_dict(state_dict), strict=strict)
    return checkpoint


def save_training_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, best_psnr
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )
