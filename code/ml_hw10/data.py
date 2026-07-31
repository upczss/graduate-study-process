from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def natural_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    return path.stem.rstrip("0123456789"), int(match.group(1)) if match else -1


class AttackDataset(Dataset):
    """Load the 200 official HW10 PNG files in raw [0, 1] pixel space."""

    def __init__(self, root: Path, limit: int | None = None) -> None:
        self.root = Path(root)
        self.samples: list[tuple[Path, int, Path]] = []
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.root}")

        for label, class_name in enumerate(CLASSES):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            files = sorted(class_dir.glob("*.png"), key=natural_key)
            if len(files) != 20 and limit is None:
                raise ValueError(
                    f"Expected 20 PNG files in {class_dir}, found {len(files)}"
                )
            self.samples.extend(
                (path, label, Path(class_name) / path.name) for path in files
            )

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, relative_path = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (32, 32):
                raise ValueError(f"Expected 32x32 image, got {image.size}: {path}")
            tensor = pil_to_tensor(image).float().div_(255.0)
        return tensor, label, relative_path.as_posix()


def load_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_official_data(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in CLASSES:
        class_dir = Path(root) / class_name
        files = sorted(class_dir.glob("*.png"), key=natural_key)
        counts[class_name] = len(files)
        for path in files:
            with Image.open(path) as image:
                if image.size != (32, 32) or image.mode not in ("RGB", "RGBA"):
                    raise ValueError(
                        f"Invalid image {path}: size={image.size}, mode={image.mode}"
                    )
    if sum(counts.values()) != 200 or any(value != 20 for value in counts.values()):
        raise ValueError(f"Expected 10 x 20 = 200 images, got: {counts}")
    return counts
