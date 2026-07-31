from __future__ import annotations

from collections.abc import Sequence
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
import torch.nn.functional as F

from common import (
    class_name,
    denormalize_tensor,
    normalize_array,
    normalize_tensor,
    predict_food,
    read_rgb,
    save_heatmap_grid,
)
from models import FoodClassifier, find_conv_layer


def mark_boundaries(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Draw yellow borders without requiring scikit-image."""
    result = np.asarray(image).copy()
    boundary = np.zeros(segments.shape, dtype=bool)
    boundary[1:, :] |= segments[1:, :] != segments[:-1, :]
    boundary[:, 1:] |= segments[:, 1:] != segments[:, :-1]
    result[boundary] = np.array((1.0, 1.0, 0.0), dtype=result.dtype)
    return result


def _target_classes(model: FoodClassifier, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(batch).argmax(dim=-1)


def saliency_maps(
    model: FoodClassifier, batch: torch.Tensor, targets: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch.detach().clone().requires_grad_(True)
    if targets is None:
        targets = _target_classes(model, inputs)
    scores = model(inputs).gather(1, targets[:, None]).sum()
    model.zero_grad(set_to_none=True)
    scores.backward()
    maps = inputs.grad.detach().abs().amax(dim=1)
    maps /= maps.flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return maps, targets


def smoothgrad_maps(
    model: FoodClassifier,
    batch: torch.Tensor,
    samples: int = 30,
    noise_std: float = 0.15,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if targets is None:
        targets = _target_classes(model, batch)
    total = torch.zeros(
        (batch.size(0), batch.size(2), batch.size(3)), device=batch.device
    )
    data_range = batch.flatten(1).amax(dim=1) - batch.flatten(1).amin(dim=1)
    for _ in range(samples):
        noise = torch.randn_like(batch) * data_range.view(-1, 1, 1, 1) * noise_std
        maps, _ = saliency_maps(model, batch + noise, targets)
        total += maps
    total /= samples
    total /= total.flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return total, targets


def integrated_gradients(
    model: FoodClassifier,
    batch: torch.Tensor,
    steps: int = 50,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if targets is None:
        targets = _target_classes(model, batch)
    baseline = torch.zeros_like(batch)
    gradient_sum = torch.zeros_like(batch)
    for alpha in torch.linspace(0, 1, steps, device=batch.device):
        point = (baseline + alpha * (batch - baseline)).detach().requires_grad_(True)
        score = model(point).gather(1, targets[:, None]).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        gradient_sum += point.grad.detach()
    attribution = (batch - baseline) * gradient_sum / steps
    maps = attribution.abs().sum(dim=1)
    maps /= maps.flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return maps, targets


def run_lime(
    model: FoodClassifier,
    images: Sequence[np.ndarray],
    device: torch.device,
    output_path: Path,
    num_samples: int = 500,
) -> None:
    try:
        from lime import lime_image
    except ImportError as exc:
        raise RuntimeError("LIME is not installed. Run: pip install lime") from exc

    def classifier(values: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(values), 64):
            tensor = normalize_array(values[start : start + 64]).to(device)
            with torch.no_grad():
                outputs.append(model(tensor).softmax(dim=-1).cpu().numpy())
        return np.concatenate(outputs)

    explainer = lime_image.LimeImageExplainer(random_state=42)
    figure, axes = plt.subplots(len(images), 2, figsize=(8, max(3, 3 * len(images))), squeeze=False)
    for row, image in enumerate(images):
        explanation = explainer.explain_instance(
            image,
            classifier,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
        )
        label = explanation.top_labels[0]
        explained, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=8,
            hide_rest=False,
        )
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"prediction: {class_name(label)}")
        axes[row, 1].imshow(mark_boundaries(explained, mask))
        axes[row, 1].set_title("LIME positive regions")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def filter_feature_maps(
    model: FoodClassifier,
    batch: torch.Tensor,
    layer_name: str,
    filter_index: int,
) -> torch.Tensor:
    layer = find_conv_layer(model, layer_name)
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    features = captured[0]
    if not 0 <= filter_index < features.size(1):
        raise ValueError(f"Filter index must be between 0 and {features.size(1) - 1}")
    maps = features[:, filter_index].relu()
    maps = F.interpolate(
        maps[:, None], size=batch.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    maps /= maps.flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return maps


def maximize_filter(
    model: FoodClassifier,
    layer_name: str,
    filter_index: int,
    image_size: int,
    device: torch.device,
    steps: int = 100,
    learning_rate: float = 0.08,
) -> np.ndarray:
    layer = find_conv_layer(model, layer_name)
    activation: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        activation.clear()
        activation.append(output)

    handle = layer.register_forward_hook(hook)
    pixels = torch.rand(1, 3, image_size, image_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pixels], lr=learning_rate)
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model(normalize_tensor(pixels.clamp(0, 1)))
            if not 0 <= filter_index < activation[0].size(1):
                raise ValueError(
                    f"Filter index must be between 0 and {activation[0].size(1) - 1}"
                )
            loss = -activation[0][:, filter_index].mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0, 1)
    finally:
        handle.remove()
    return pixels.detach()[0].permute(1, 2, 0).cpu().numpy()


def run_cnn_explanations(
    model: FoodClassifier,
    image_paths: Sequence[Path],
    image_size: int,
    device: torch.device,
    output_dir: Path,
    methods: Sequence[str],
    layer_name: str,
    filter_index: int,
    smooth_samples: int,
    ig_steps: int,
    lime_samples: int,
) -> None:
    originals = [read_rgb(path, image_size) for path in image_paths]
    batch = normalize_array(np.stack(originals)).to(device)
    probabilities, predictions = predict_food(model, batch)
    titles = [
        f"{path.name} | {class_name(int(label))} ({float(probability):.3f})"
        for path, label, probability in zip(image_paths, predictions, probabilities)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    if "saliency" in methods:
        maps, _ = saliency_maps(model, batch, predictions)
        save_heatmap_grid(originals, list(maps.cpu().numpy()), titles, output_dir / "saliency.png")
    if "smoothgrad" in methods:
        maps, _ = smoothgrad_maps(model, batch, smooth_samples, targets=predictions)
        save_heatmap_grid(originals, list(maps.cpu().numpy()), titles, output_dir / "smoothgrad.png")
    if "integrated-gradients" in methods:
        maps, _ = integrated_gradients(model, batch, ig_steps, predictions)
        save_heatmap_grid(
            originals, list(maps.cpu().numpy()), titles, output_dir / "integrated_gradients.png"
        )
    if "filter" in methods:
        maps = filter_feature_maps(model, batch, layer_name, filter_index)
        save_heatmap_grid(
            originals,
            list(maps.cpu().numpy()),
            titles,
            output_dir / f"filter_{layer_name.replace('.', '_')}_{filter_index}.png",
            cmap="viridis",
        )
        pattern = maximize_filter(
            model, layer_name, filter_index, image_size, device
        )
        plt.imsave(
            output_dir / f"filter_maximization_{layer_name.replace('.', '_')}_{filter_index}.png",
            pattern,
        )
    if "lime" in methods:
        run_lime(model, originals, device, output_dir / "lime.png", lime_samples)

    lines = ["file,predicted_class,probability"]
    lines.extend(
        f"{path.name},{class_name(int(label))},{float(probability):.6f}"
        for path, label, probability in zip(image_paths, predictions, probabilities)
    )
    (output_dir / "predictions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
