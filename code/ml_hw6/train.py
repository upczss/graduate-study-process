import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import save_image

from dataset import build_dataloader
from losses import (
    dcgan_discriminator_loss,
    dcgan_generator_loss,
    gradient_penalty,
    wasserstein_critic_loss,
    wasserstein_generator_loss,
)
from models import Discriminator, Generator


MODES = ("dcgan", "wgan", "wgan_gp")
PROJECT_CODE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_CODE_DIR / "data" / "hw6" / "faces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN, WGAN, or WGAN-GP.")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to faces/ or an ImageFolder root (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--critic-iterations", type=int, default=5)
    parser.add_argument("--clip-value", type=float, default=0.01)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizers(args, generator, discriminator):
    if args.mode == "dcgan":
        lr = args.lr or 2e-4
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    elif args.mode == "wgan":
        lr = args.lr or 5e-5
        optimizer_g = optim.RMSprop(generator.parameters(), lr=lr)
        optimizer_d = optim.RMSprop(discriminator.parameters(), lr=lr)
    else:
        lr = args.lr or 1e-4
        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    return optimizer_g, optimizer_d


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
) -> None:
    torch.save(
        {
            "mode": args.mode,
            "epoch": epoch,
            "global_step": global_step,
            "latent_dim": args.latent_dim,
            "generator_features": args.generator_features,
            "discriminator_features": args.discriminator_features,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, generator, discriminator, optimizer_g, optimizer_d, mode, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint["mode"] != mode:
        raise ValueError(
            f"Checkpoint mode is {checkpoint['mode']}, but --mode is {mode}."
        )
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    return checkpoint["epoch"] + 1, checkpoint["global_step"]


def train() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.critic_iterations < 1:
        raise ValueError("--critic-iterations must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir) / args.mode
    sample_dir = run_dir / "samples"
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataloader = build_dataloader(
        args.data_dir, args.batch_size, num_workers=args.num_workers
    )
    generator = Generator(args.latent_dim, args.generator_features).to(device)
    discriminator = Discriminator(args.mode, args.discriminator_features).to(device)
    optimizer_g, optimizer_d = build_optimizers(args, generator, discriminator)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    start_epoch, global_step = 1, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            args.mode,
            device,
        )

    print(f"Mode: {args.mode} | Device: {device} | Images: {len(dataloader.dataset)}")
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()

        for batch_index, (real_images, _) in enumerate(dataloader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # Update discriminator/critic.
            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise).detach()
            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images)

            if args.mode == "dcgan":
                loss_d = dcgan_discriminator_loss(real_scores, fake_scores, criterion)
            else:
                loss_d = wasserstein_critic_loss(real_scores, fake_scores)
                if args.mode == "wgan_gp":
                    penalty = gradient_penalty(
                        discriminator, real_images, fake_images, device
                    )
                    loss_d = loss_d + args.gp_lambda * penalty

            loss_d.backward()
            optimizer_d.step()

            if args.mode == "wgan":
                with torch.no_grad():
                    for parameter in discriminator.parameters():
                        parameter.clamp_(-args.clip_value, args.clip_value)

            # DCGAN updates G every batch; Wasserstein variants update after
            # several critic steps.
            update_generator = (
                args.mode == "dcgan"
                or global_step % args.critic_iterations == 0
            )
            loss_g_value = float("nan")
            if update_generator:
                optimizer_g.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                generated = generator(noise)
                generated_scores = discriminator(generated)
                loss_g = (
                    dcgan_generator_loss(generated_scores, criterion)
                    if args.mode == "dcgan"
                    else wasserstein_generator_loss(generated_scores)
                )
                loss_g.backward()
                optimizer_g.step()
                loss_g_value = loss_g.item()

            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} "
                    f"Batch {batch_index:04d}/{len(dataloader):04d} "
                    f"D: {loss_d.item():.4f} G: {loss_g_value:.4f}"
                )

            if global_step % args.sample_every == 0:
                generator.eval()
                with torch.no_grad():
                    samples = generator(fixed_noise)
                save_image(
                    samples,
                    sample_dir / f"step_{global_step:07d}.jpg",
                    normalize=True,
                    value_range=(-1, 1),
                    nrow=8,
                )
                generator.train()

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                args,
                epoch,
                global_step,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                args,
                epoch,
                global_step,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
            )


if __name__ == "__main__":
    train()
