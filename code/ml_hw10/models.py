from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


CIFAR10_MEAN = (0.491, 0.482, 0.447)
CIFAR10_STD = (0.202, 0.199, 0.201)


class NormalizedModel(nn.Module):
    """Accept raw [0, 1] images and normalize them before the proxy model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.mean) / self.std)


class EnsembleModel(nn.Module):
    def __init__(self, models: Sequence[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one proxy model is required")
        self.models = nn.ModuleList(models)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = [model(images) for model in self.models]
        return torch.stack(logits, dim=0).mean(dim=0)


def load_proxy_models(names: Sequence[str], device: torch.device) -> nn.Module:
    try:
        from pytorchcv.model_provider import get_model
    except ImportError as exc:
        raise RuntimeError(
            "pytorchcv is not installed. Run: python -m pip install pytorchcv"
        ) from exc

    proxies: list[nn.Module] = []
    for name in names:
        print(f"Loading proxy model: {name}")
        base = get_model(name, pretrained=True)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        proxies.append(NormalizedModel(base).to(device).eval())

    if len(proxies) == 1:
        return proxies[0]
    return EnsembleModel(proxies).to(device).eval()
