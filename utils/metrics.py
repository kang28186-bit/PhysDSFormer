import torch
import torch.nn.functional as F

from losses.multicolor import structural_similarity


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.value = float(value)
        self.total += self.value * count
        self.count += count
        self.average = self.total / max(self.count, 1)


def batch_psnr(prediction, target):
    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    return (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean()


def batch_ssim(prediction, target):
    return structural_similarity(prediction, target)

