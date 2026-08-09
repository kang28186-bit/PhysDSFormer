import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from losses import MultiColorSpaceStructuralLoss  # noqa: E402
from models import physdsformer_tiny  # noqa: E402


def main():
    torch.manual_seed(7)
    model = physdsformer_tiny()
    source = torch.randn(1, 3, 64, 64).clamp(-1.0, 1.0)
    target = torch.randn(1, 3, 64, 64).clamp(-1.0, 1.0)
    prediction = model(source)
    criterion = MultiColorSpaceStructuralLoss()
    loss = criterion(prediction, target)
    loss.backward()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 51554
    assert prediction.shape == source.shape
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)

    for value, target_value in ((-1.0, -1.0), (2.0, 0.0)):
        probe = torch.full((1, 3, 16, 16), value, requires_grad=True)
        probe_target = torch.full_like(probe, target_value)
        probe_loss = criterion(probe, probe_target)
        probe_loss.backward()
        assert torch.isfinite(probe_loss)
        assert torch.isfinite(probe.grad).all()
    assert probe.grad.abs().max() > 0

    print("parameters={}".format(parameters))
    print("output_shape={}".format(tuple(prediction.shape)))
    print("loss={:.6f}".format(loss.item()))
    print("gradient_checks=passed")


if __name__ == "__main__":
    main()
