from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class FCNAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        input_dim = 64 * 64 * 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_dim),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        shape = images.shape
        latent = self.encoder(images.flatten(1))
        return self.decoder(latent).reshape(shape)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        return self(images)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            self._down(3, 32),
            self._down(32, 64),
            self._down(64, 128),
            self._down(128, 256),
        )
        self.encoder_linear = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            self._up(256, 128),
            self._up(128, 64),
            self._up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _down(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _up(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder_linear(self.encoder_conv(images).flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_linear(latent).reshape(-1, 256, 4, 4)
        return self.decoder(hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        return self(images)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            ConvAutoencoder._down(3, 32),
            ConvAutoencoder._down(32, 64),
            ConvAutoencoder._down(64, 128),
            ConvAutoencoder._down(128, 256),
        )
        self.mean = nn.Linear(256 * 4 * 4, latent_dim)
        self.log_variance = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            ConvAutoencoder._up(256, 128),
            ConvAutoencoder._up(128, 64),
            ConvAutoencoder._up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_conv(images).flatten(1)
        return self.mean(hidden), self.log_variance(hidden)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_linear(latent).reshape(-1, 256, 4, 4)
        return self.decoder(hidden)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        reconstruction = self.decode(self.reparameterize(mean, log_variance))
        return reconstruction, mean, log_variance

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        # Using the mean instead of sampling makes anomaly scores deterministic.
        mean, _ = self.encode(images)
        return self.decode(mean)


def build_model(model_type: str, latent_dim: int) -> nn.Module:
    model_classes = {
        "fcn": FCNAutoencoder,
        "cnn": ConvAutoencoder,
        "vae": VariationalAutoencoder,
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_classes[model_type](latent_dim=latent_dim)


def calculate_loss(
    model_type: str,
    output: Any,
    target: torch.Tensor,
    vae_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_type != "vae":
        reconstruction_loss = F.mse_loss(output, target)
        return reconstruction_loss, reconstruction_loss

    reconstruction, mean, log_variance = output
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl_loss = -0.5 * torch.mean(
        1.0 + log_variance - mean.pow(2) - log_variance.exp()
    )
    return reconstruction_loss + vae_beta * kl_loss, reconstruction_loss

