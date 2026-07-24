import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from data import NpyImageDataset, split_indices
from models import build_model, calculate_loss


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CODE_DIR.parent / "data" / "hw8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an autoencoder for HW8.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(CODE_DIR / "outputs"))
    parser.add_argument("--model", choices=("fcn", "cnn", "vae"), default="fcn")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--vae-beta", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Use only the first N selected images for a quick test.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    args,
    epoch: int,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_type": args.model,
            "latent_dim": args.latent_dim,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "args": vars(args),
        },
        path,
    )


def validate(model, model_type, loader, device, use_fp16, vae_beta):
    model.eval()
    total_loss = 0.0
    total_images = 0
    preview_input = preview_output = None
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                output = model(images)
                _, reconstruction_loss = calculate_loss(
                    model_type, output, images, vae_beta
                )
                reconstruction = (
                    output[0] if model_type == "vae" else output
                )
            total_loss += reconstruction_loss.item() * images.size(0)
            total_images += images.size(0)
            if preview_input is None:
                preview_input = images[:8].float().cpu()
                preview_output = reconstruction[:8].float().cpu()
    return total_loss / total_images, preview_input, preview_output


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    data_path = Path(args.data_dir) / "trainingset.npy"
    output_dir = Path(args.output_dir) / args.model
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    image_count = len(np.load(data_path, mmap_mode="r"))
    train_indices, validation_indices = split_indices(
        image_count, args.validation_ratio, args.seed
    )
    if args.limit_train:
        train_indices = train_indices[: args.limit_train]
        validation_indices = validation_indices[: max(32, args.limit_train // 5)]

    train_dataset = NpyImageDataset(data_path, train_indices, augment=True)
    validation_dataset = NpyImageDataset(
        data_path, validation_indices, augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    best_validation_loss = float("inf")

    print(
        f"Model: {args.model} | Device: {device} | FP16: {use_fp16}\n"
        f"Train images: {len(train_dataset)} | "
        f"Validation images: {len(validation_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_images = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for clean_images in progress:
            clean_images = clean_images.to(device, non_blocking=True)
            if args.noise_std > 0:
                noisy_images = (
                    clean_images
                    + torch.randn_like(clean_images) * args.noise_std
                ).clamp(-1.0, 1.0)
            else:
                noisy_images = clean_images

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                output = model(noisy_images)
                loss, _ = calculate_loss(
                    args.model, output, clean_images, args.vae_beta
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * clean_images.size(0)
            total_images += clean_images.size(0)
            progress.set_postfix(loss=f"{total_loss / total_images:.5f}")

        validation_loss, preview_input, preview_output = validate(
            model,
            args.model,
            validation_loader,
            device,
            use_fp16,
            args.vae_beta,
        )
        scheduler.step()
        print(
            f"Epoch {epoch}: train loss={total_loss / total_images:.6f}, "
            f"validation reconstruction={validation_loss:.6f}"
        )

        preview = torch.cat((preview_input, preview_output), dim=0)
        save_image(
            preview,
            preview_dir / f"epoch_{epoch:03d}.jpg",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            args,
            epoch,
            validation_loss,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                args,
                epoch,
                validation_loss,
            )
            print(f"Saved new best model: {checkpoint_dir / 'best.pt'}")

    print(f"Training complete. Best validation loss: {best_validation_loss:.6f}")


if __name__ == "__main__":
    main()
