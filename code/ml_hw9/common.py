from __future__ import annotations

import random
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import FOOD_CLASSES, FoodClassifier


ROOT = Path(__file__).resolve().parents[1]
HW9_DIR = Path(__file__).resolve().parent
DEFAULT_HW3_DATA = ROOT / "data" / "hw3"
DEFAULT_HW3_MODEL = ROOT / "ml_hw3" / "best_model.pt"
DEFAULT_HW7_MODEL = ROOT / "ml_hw7" / "outputs" / "best_model"
DEFAULT_OUTPUT = HW9_DIR / "outputs"

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_food_model(model_path: Path, device: torch.device) -> tuple[FoodClassifier, int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"HW3 model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    image_size = int(checkpoint.get("image_size", 128))
    model = FoodClassifier()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, image_size


def list_food_images(folder: Path) -> list[Path]:
    images = sorted(
        folder.glob("*.jpg"),
        key=lambda path: tuple(
            int(part) if part.isdigit() else part for part in path.stem.split("_")
        ),
    )
    if not images:
        raise FileNotFoundError(f"No JPG images found in {folder}")
    return images


def read_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


def normalize_array(images: np.ndarray) -> torch.Tensor:
    values = (images - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(values.transpose(0, 3, 1, 2)).float()


def normalize_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def denormalize_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def predict_food(
    model: FoodClassifier, normalized: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model(normalized)
        probabilities = logits.softmax(dim=-1)
    return probabilities.max(dim=-1)


def save_heatmap_grid(
    originals: list[np.ndarray],
    heatmaps: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    cmap: str = "magma",
) -> None:
    count = len(originals)
    figure, axes = plt.subplots(count, 2, figsize=(8, max(3, 3 * count)), squeeze=False)
    for row, (image, heatmap, title) in enumerate(zip(originals, heatmaps, titles)):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(title)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(heatmap, cmap=cmap, alpha=0.55)
        axes[row, 1].set_title("explanation")
        axes[row, 1].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def class_name(index: int) -> str:
    return FOOD_CLASSES[index] if 0 <= index < len(FOOD_CLASSES) else str(index)
