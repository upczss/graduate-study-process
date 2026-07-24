import argparse
import tarfile
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW6 submission images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--archive", default="images.tgz")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing JPG files in output-dir before generation.",
    )
    return parser.parse_args()


def generate() -> None:
    args = parse_args()
    if args.num_images < 1 or args.batch_size < 1:
        raise ValueError("--num-images and --batch-size must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    latent_dim = checkpoint.get("latent_dim", 128)
    generator_features = checkpoint.get("generator_features", 64)
    generator = Generator(latent_dim, generator_features).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jpg"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains JPG files. Use --overwrite to replace them."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    generated_count = 0
    with torch.inference_mode():
        while generated_count < args.num_images:
            current_batch = min(args.batch_size, args.num_images - generated_count)
            noise = torch.randn(current_batch, latent_dim, 1, 1, device=device)
            images = generator(noise).cpu().add(1).div(2).clamp(0, 1)
            for image in images:
                generated_count += 1
                to_pil_image(image).save(
                    output_dir / f"{generated_count}.jpg",
                    format="JPEG",
                    quality=args.jpeg_quality,
                    optimize=True,
                )

    archive_path = Path(args.archive)
    with tarfile.open(archive_path, "w:gz") as archive:
        for index in range(1, args.num_images + 1):
            image_path = output_dir / f"{index}.jpg"
            archive.add(image_path, arcname=image_path.name)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Generated {args.num_images} images in {output_dir}")
    print(f"Created {archive_path} ({size_mb:.2f} MB)")
    if size_mb >= 2:
        print(
            "Warning: the original HW6 limit was 2 MB. Regenerate with a lower "
            "--jpeg-quality, for example 70."
        )


if __name__ == "__main__":
    generate()
