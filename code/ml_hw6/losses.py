import torch
from torch import autograd, nn


def dcgan_discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    real_targets = torch.ones_like(real_scores)
    fake_targets = torch.zeros_like(fake_scores)
    return criterion(real_scores, real_targets) + criterion(fake_scores, fake_targets)


def dcgan_generator_loss(
    fake_scores: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return criterion(fake_scores, torch.ones_like(fake_scores))


def wasserstein_critic_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    return fake_scores.mean() - real_scores.mean()


def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1.0 - alpha) * fake_images
    interpolated.requires_grad_(True)

    scores = critic(interpolated)
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

