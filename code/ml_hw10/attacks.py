from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_linf(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    adversarial = torch.maximum(
        torch.minimum(adversarial, original + epsilon), original - epsilon
    )
    return adversarial.clamp(0.0, 1.0)


def input_diversity(
    images: torch.Tensor,
    probability: float = 0.7,
    max_resize: int = 36,
) -> torch.Tensor:
    """DIM: random resize + random zero-padding + resize back to 32x32."""
    if probability <= 0 or torch.rand((), device=images.device) > probability:
        return images
    image_size = images.shape[-1]
    resize = int(
        torch.randint(image_size, max_resize + 1, (), device=images.device).item()
    )
    resized = F.interpolate(
        images, size=(resize, resize), mode="bilinear", align_corners=False
    )
    remaining = max_resize - resize
    top = int(torch.randint(0, remaining + 1, (), device=images.device).item())
    left = int(torch.randint(0, remaining + 1, (), device=images.device).item())
    padded = F.pad(
        resized,
        (left, remaining - left, top, remaining - top),
        mode="constant",
        value=0.0,
    )
    return F.interpolate(
        padded, size=(image_size, image_size), mode="bilinear", align_corners=False
    )


def _gradient(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    use_dim: bool,
    dim_probability: float,
    dim_resize: int,
) -> torch.Tensor:
    inputs = images.detach().requires_grad_(True)
    model_inputs = (
        input_diversity(inputs, dim_probability, dim_resize) if use_dim else inputs
    )
    loss = F.cross_entropy(model(model_inputs), labels)
    gradient = torch.autograd.grad(loss, inputs, only_inputs=True)[0]
    return gradient.detach()


def fgsm(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    **_: object,
) -> torch.Tensor:
    gradient = _gradient(model, images, labels, False, 0.0, 36)
    return project_linf(images + epsilon * gradient.sign(), images, epsilon).detach()


def iterative_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 0.8 / 255,
    steps: int = 20,
    momentum_decay: float = 0.0,
    use_dim: bool = False,
    dim_probability: float = 0.7,
    dim_resize: int = 36,
    random_start: bool = False,
) -> torch.Tensor:
    original = images.detach()
    if random_start:
        adversarial = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        adversarial = project_linf(adversarial, original, epsilon)
    else:
        adversarial = original.clone()
    momentum = torch.zeros_like(adversarial)

    for _ in range(steps):
        gradient = _gradient(
            model,
            adversarial,
            labels,
            use_dim,
            dim_probability,
            dim_resize,
        )
        if momentum_decay > 0:
            scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            momentum = momentum_decay * momentum + gradient / scale
            direction = momentum
        else:
            direction = gradient
        adversarial = adversarial + alpha * direction.sign()
        adversarial = project_linf(adversarial, original, epsilon).detach()
    return adversarial


def run_attack(
    name: str,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    if name == "fgsm":
        return fgsm(model, images, labels, **kwargs)
    if name == "ifgsm":
        return iterative_attack(
            model, images, labels, momentum_decay=0.0, use_dim=False, **kwargs
        )
    if name == "mifgsm":
        return iterative_attack(
            model, images, labels, momentum_decay=1.0, use_dim=False, **kwargs
        )
    if name == "dim-mifgsm":
        return iterative_attack(
            model, images, labels, momentum_decay=1.0, use_dim=True, **kwargs
        )
    raise ValueError(f"Unknown attack: {name}")
