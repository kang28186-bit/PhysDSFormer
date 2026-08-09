import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_lab(rgb):
    """Convert an sRGB tensor in [0, 1] to CIE LAB (D65), differentiably."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected a BCHW RGB tensor")

    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )
    matrix = linear.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    white = xyz.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    xyz = xyz / white

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f_xyz = torch.where(
        xyz > epsilon,
        torch.pow(xyz.clamp_min(epsilon), 1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0,
    )
    fx, fy, fz = torch.unbind(f_xyz, dim=1)
    lightness = 116.0 * fy - 16.0
    green_red = 500.0 * (fx - fy)
    blue_yellow = 200.0 * (fy - fz)
    return torch.stack((lightness, green_red, blue_yellow), dim=1)


def _soft_histogram(values, levels, epsilon=1e-6):
    """Build a per-image triangular soft histogram for values in [0, 1]."""
    values = values.clamp(0.0, 1.0).flatten(1)
    position = values * (levels - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(levels - 1)
    upper_weight = position - lower.to(position.dtype)
    lower_weight = 1.0 - upper_weight

    histogram = values.new_zeros(values.shape[0], levels)
    histogram.scatter_add_(1, lower, lower_weight)
    histogram.scatter_add_(1, upper, upper_weight)
    histogram = histogram + epsilon
    return histogram / histogram.sum(dim=1, keepdim=True)


def _soft_quantized_cross_entropy(prediction, target, levels):
    """Cross entropy between image-level differentiable color histograms."""
    pred_distribution = _soft_histogram(prediction, levels)
    target_distribution = _soft_histogram(target, levels).detach()
    return -(
        target_distribution * pred_distribution.clamp_min(1e-12).log()
    ).sum(dim=1).mean()


def _gaussian_filter(image, window_size, sigma):
    channels = image.shape[1]
    coordinates = torch.arange(
        window_size, device=image.device, dtype=image.dtype
    ) - (window_size - 1) / 2.0
    gaussian = torch.exp(-coordinates.square() / (2.0 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    kernel = kernel.view(1, 1, window_size, window_size).expand(
        channels, 1, window_size, window_size
    )
    padding = window_size // 2
    if padding:
        mode = "reflect" if min(image.shape[-2:]) > padding else "replicate"
        image = F.pad(image, (padding, padding, padding, padding), mode=mode)
    return F.conv2d(image, kernel, groups=channels)


def structural_similarity(prediction, target, window_size=11, sigma=1.5):
    """Differentiable single-scale SSIM with a Gaussian local window."""
    window_size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)

    mu_pred = _gaussian_filter(prediction, window_size, sigma)
    mu_target = _gaussian_filter(target, window_size, sigma)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_pred * mu_target

    sigma_pred = (
        _gaussian_filter(prediction.square(), window_size, sigma) - mu_pred_sq
    )
    sigma_target = (
        _gaussian_filter(target.square(), window_size, sigma) - mu_target_sq
    )
    sigma_cross = (
        _gaussian_filter(prediction * target, window_size, sigma) - mu_product
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (
        sigma_pred + sigma_target + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean()


class MultiColorSpaceStructuralLoss(nn.Module):
    """RGB, LAB, LCH and structural supervision described in the paper."""

    def __init__(
        self,
        rgb_weight=0.001,
        lab_weight=100.0,
        lch_weight=0.1,
        ssim_weight=1.0,
        quantization_levels=32,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.lab_weight = lab_weight
        self.lch_weight = lch_weight
        self.ssim_weight = ssim_weight
        self.quantization_levels = quantization_levels

    def forward(self, prediction, target, return_components=False):
        prediction_rgb_raw = prediction.float() * 0.5 + 0.5
        prediction_rgb = prediction_rgb_raw.clamp(0.0, 1.0)
        target_rgb = (target.float() * 0.5 + 0.5).clamp(0.0, 1.0)

        # Keep this term unclamped so it pulls out-of-range predictions back.
        rgb_loss = F.mse_loss(prediction_rgb_raw, target_rgb)

        pred_lab = rgb_to_lab(prediction_rgb)
        target_lab = rgb_to_lab(target_rgb)
        pred_l = pred_lab[:, 0] / 100.0
        target_l = target_lab[:, 0] / 100.0
        pred_a = ((pred_lab[:, 1] + 128.0) / 255.0).clamp(0.0, 1.0)
        target_a = ((target_lab[:, 1] + 128.0) / 255.0).clamp(0.0, 1.0)
        pred_b = ((pred_lab[:, 2] + 128.0) / 255.0).clamp(0.0, 1.0)
        target_b = ((target_lab[:, 2] + 128.0) / 255.0).clamp(0.0, 1.0)
        lab_loss = F.mse_loss(pred_l, target_l)
        lab_loss = lab_loss + _soft_quantized_cross_entropy(
            pred_a, target_a, self.quantization_levels
        )
        lab_loss = lab_loss + _soft_quantized_cross_entropy(
            pred_b, target_b, self.quantization_levels
        )

        lch_lightness = _soft_quantized_cross_entropy(
            pred_l,
            target_l,
            self.quantization_levels,
        )
        chroma_epsilon = 1e-6
        pred_chroma = torch.sqrt(
            pred_lab[:, 1].square()
            + pred_lab[:, 2].square()
            + chroma_epsilon ** 2
        )
        target_chroma = torch.sqrt(
            target_lab[:, 1].square()
            + target_lab[:, 2].square()
            + chroma_epsilon ** 2
        )
        chroma_scale = 128.0 * (2.0 ** 0.5)
        chroma_loss = F.mse_loss(
            (pred_chroma / chroma_scale).clamp(0.0, 1.0),
            (target_chroma / chroma_scale).clamp(0.0, 1.0),
        )
        # Cosine hue distance. Low-chroma targets carry little reliable hue.
        hue_dot = (
            pred_lab[:, 1] * target_lab[:, 1]
            + pred_lab[:, 2] * target_lab[:, 2]
        )
        hue_cosine = (hue_dot / (pred_chroma * target_chroma)).clamp(-1.0, 1.0)
        target_chroma_for_weight = torch.sqrt(
            target_lab[:, 1].square() + target_lab[:, 2].square()
        )
        hue_weight = (target_chroma_for_weight / 5.0).clamp(0.0, 1.0).detach()
        hue_loss = (hue_weight * (1.0 - hue_cosine)).sum() / (
            hue_weight.sum() + 1e-6
        )
        lch_loss = lch_lightness + chroma_loss + hue_loss

        ssim_loss = 1.0 - structural_similarity(prediction_rgb, target_rgb)
        total = (
            self.rgb_weight * rgb_loss
            + self.lab_weight * lab_loss
            + self.lch_weight * lch_loss
            + self.ssim_weight * ssim_loss
        )

        if return_components:
            return total, {
                "rgb": rgb_loss.detach(),
                "lab": lab_loss.detach(),
                "lch": lch_loss.detach(),
                "ssim": ssim_loss.detach(),
            }
        return total
