import torch
from torch import nn


def initialize_weights(module: nn.Module) -> None:
    """DCGAN-style parameter initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


class Generator(nn.Module):
    """Map latent vectors (N, latent_dim, 1, 1) to 64x64 RGB images."""

    def __init__(self, latent_dim: int = 128, features: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            self._block(latent_dim, features * 8, 4, 1, 0),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(initialize_weights)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """
    Score 64x64 RGB images.

    DCGAN applies sigmoid because it uses binary cross entropy. WGAN and
    WGAN-GP return an unrestricted critic score.
    """

    def __init__(self, mode: str = "dcgan", features: int = 64) -> None:
        super().__init__()
        if mode not in {"dcgan", "wgan", "wgan_gp"}:
            raise ValueError(f"Unsupported mode: {mode}")

        layers: list[nn.Module] = [
            nn.Conv2d(3, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2, use_batch_norm=mode == "dcgan"),
            self._block(features * 2, features * 4, use_batch_norm=mode == "dcgan"),
            self._block(features * 4, features * 8, use_batch_norm=mode == "dcgan"),
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
        ]
        if mode == "dcgan":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self.apply(initialize_weights)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        ]
        # A Wasserstein critic should not couple samples through batch
        # statistics, especially when computing WGAN-GP per-sample gradients.
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).reshape(-1)
