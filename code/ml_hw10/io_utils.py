from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import CLASSES, load_uint8


def tensor_to_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    path = Path(path).resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        if path.name in ("", ".", "..") or len(path.parts) < 4:
            raise ValueError(f"Refusing to remove unsafe output path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def save_adversarial_batch(
    images: torch.Tensor,
    relative_paths: list[str] | tuple[str, ...],
    output_dir: Path,
) -> None:
    arrays = tensor_to_uint8(images)
    for array, relative in zip(arrays, relative_paths):
        destination = output_dir / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(destination, format="PNG")


def verify_linf(
    original_dir: Path,
    adversarial_dir: Path,
    expected_count: int,
    epsilon_pixels: int = 8,
) -> dict[str, float | int]:
    differences: list[int] = []
    files = sorted(adversarial_dir.glob("*/*.png"))
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} adversarial PNG files, got {len(files)}")
    for adversarial_path in files:
        relative = adversarial_path.relative_to(adversarial_dir)
        original_path = original_dir / relative
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original image: {original_path}")
        original = load_uint8(original_path).astype(np.int16)
        adversarial = load_uint8(adversarial_path).astype(np.int16)
        differences.append(int(np.abs(adversarial - original).max()))
    maximum = max(differences, default=0)
    if maximum > epsilon_pixels:
        raise ValueError(
            f"L-infinity constraint violated: maximum={maximum}, allowed={epsilon_pixels}"
        )
    return {
        "image_count": len(files),
        "max_linf_pixels": maximum,
        "mean_per_image_linf_pixels": float(np.mean(differences)),
    }


def create_submission_archive(source_dir: Path, archive_path: Path) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for class_name in CLASSES:
            class_dir = source_dir / class_name
            if class_dir.is_dir():
                archive.add(class_dir, arcname=class_name)
    size = archive_path.stat().st_size
    if size >= 2 * 1024 * 1024:
        raise ValueError(f"Submission archive must be under 2 MB, got {size / 1024**2:.3f} MB")
    return size


def jpeg_compress(image: Image.Image, compression: int = 70) -> Image.Image:
    if not 0 <= compression <= 100:
        raise ValueError("compression must be between 0 and 100")
    quality = max(1, 100 - compression)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
