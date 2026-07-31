from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

from attacks import run_attack
from data import CLASSES, AttackDataset, validate_official_data
from io_utils import (
    create_submission_archive,
    jpeg_compress,
    prepare_output_directory,
    save_adversarial_batch,
    verify_linf,
    write_json,
)
from models import load_proxy_models


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "data" / "hw10" / "data"
DEFAULT_OUTPUTS = SCRIPT_DIR / "outputs"
ATTACKS = ("fgsm", "ifgsm", "mifgsm", "dim-mifgsm")


def parse_model_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("At least one model name is required")
    return names


def select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> dict[str, float | int]:
    correct = 0
    total = 0
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return {"count": total, "accuracy": correct / total, "loss": loss_sum / total}


def command_check(args) -> None:
    counts = validate_official_data(args.data_dir.resolve())
    print(f"Data is valid: {sum(counts.values())} images")
    for name, count in counts.items():
        print(f"  {name}: {count}")


def make_loader(args, shuffle: bool = False) -> tuple[AttackDataset, DataLoader]:
    dataset = AttackDataset(args.data_dir.resolve(), args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
    )
    return dataset, loader


def command_attack(args) -> None:
    device = select_device(args.device)
    set_seed(args.seed)
    dataset, loader = make_loader(args)
    model = load_proxy_models(args.models, device)
    output_dir = (args.output_dir or (DEFAULT_OUTPUTS / args.attack)).resolve()
    prepare_output_directory(output_dir, args.overwrite)

    benign = evaluate(model, loader, device)
    print(f"device={device} | images={len(dataset)} | benign accuracy={benign['accuracy']:.4f}")
    started = time.perf_counter()
    adversarial_correct = 0
    total = 0
    for batch_index, (images, labels, relative_paths) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        adversarial = run_attack(
            args.attack,
            model,
            images,
            labels,
            epsilon=args.epsilon / 255,
            alpha=args.alpha / 255,
            steps=args.steps,
            dim_probability=args.dim_probability,
            dim_resize=args.dim_resize,
            random_start=args.random_start,
        )
        with torch.no_grad():
            predictions = model(adversarial).argmax(1)
        adversarial_correct += (predictions == labels).sum().item()
        total += labels.numel()
        save_adversarial_batch(adversarial, relative_paths, output_dir)
        print(f"batch {batch_index}/{len(loader)}", end="\r")
    elapsed = time.perf_counter() - started
    print()

    constraint = verify_linf(
        args.data_dir.resolve(), output_dir, len(dataset), round(args.epsilon)
    )
    metrics = {
        "attack": args.attack,
        "models": args.models,
        "epsilon_pixels": args.epsilon,
        "alpha_pixels": args.alpha,
        "steps": args.steps,
        "dim_probability": args.dim_probability,
        "benign": benign,
        "adversarial_accuracy_on_proxy": adversarial_correct / total,
        "elapsed_seconds": elapsed,
        "constraint": constraint,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"proxy adversarial accuracy={adversarial_correct / total:.4f}")
    print(f"max L-inf={constraint['max_linf_pixels']} pixels | time={elapsed:.1f}s")

    if args.pack and len(dataset) == 200:
        archive_path = args.archive or output_dir.with_suffix(".tgz")
        size = create_submission_archive(output_dir, archive_path.resolve())
        print(f"submission: {archive_path.resolve()} ({size / 1024:.1f} KB)")
    elif args.pack:
        print("Archive skipped because --limit was used; submission requires 200 images.")


def _load_raw_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div(255).unsqueeze(0).to(device)


def _prediction(model, image: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        probability = model(image).softmax(-1)[0]
    index = int(probability.argmax())
    return index, float(probability[index])


def command_defense(args) -> None:
    device = select_device(args.device)
    set_seed(args.seed)
    model = load_proxy_models([args.model], device)
    source_path = args.data_dir.resolve() / "dog" / "dog2.png"
    original = _load_raw_tensor(source_path, device)
    label = torch.tensor([CLASSES.index("dog")], device=device)
    adversarial = run_attack(
        "fgsm", model, original, label, epsilon=args.epsilon / 255
    )

    array = (
        adversarial[0].mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    )
    adversarial_image = Image.fromarray(array, mode="RGB")
    compressed_image = jpeg_compress(adversarial_image, args.compression)
    compressed = pil_to_tensor(compressed_image).float().div(255).unsqueeze(0).to(device)

    original_prediction = _prediction(model, original)
    adversarial_prediction = _prediction(model, adversarial)
    compressed_prediction = _prediction(model, compressed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adversarial_image.save(output_dir / "dog2_adversarial.png")
    compressed_image.save(output_dir / "dog2_jpeg_defense.png")

    images = [
        original[0].permute(1, 2, 0).cpu().numpy(),
        array,
        compressed_image,
    ]
    predictions = [original_prediction, adversarial_prediction, compressed_prediction]
    titles = ["benign", "FGSM", f"JPEG compression={args.compression}%"]
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for axis, image, (prediction, probability), title in zip(
        axes, images, predictions, titles
    ):
        axis.imshow(image)
        axis.set_title(f"{title}\n{CLASSES[prediction]}: {probability:.2%}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "defense_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    result = {
        "model": args.model,
        "image": str(source_path),
        "epsilon_pixels": args.epsilon,
        "jpeg_compression_percent": args.compression,
        "jpeg_quality": 100 - args.compression,
        "benign": {"class": CLASSES[original_prediction[0]], "probability": original_prediction[1]},
        "fgsm": {"class": CLASSES[adversarial_prediction[0]], "probability": adversarial_prediction[1]},
        "jpeg_defense": {"class": CLASSES[compressed_prediction[0]], "probability": compressed_prediction[1]},
    }
    write_json(output_dir / "defense_result.json", result)
    print(result)
    print(f"Defense outputs saved to {output_dir}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW10 transferable adversarial attack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate the official 200-image dataset")
    check.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    check.set_defaults(handler=command_check)

    attack = subparsers.add_parser("attack", help="Generate adversarial images")
    add_common(attack)
    attack.add_argument("--attack", choices=ATTACKS, default="fgsm")
    attack.add_argument("--models", type=parse_model_names, default=parse_model_names("resnet110_cifar10"))
    attack.add_argument("--batch-size", type=int, default=8)
    attack.add_argument("--num-workers", type=int, default=0)
    attack.add_argument("--epsilon", type=float, default=8.0)
    attack.add_argument("--alpha", type=float, default=0.8)
    attack.add_argument("--steps", type=int, default=20)
    attack.add_argument("--dim-probability", type=float, default=0.7)
    attack.add_argument("--dim-resize", type=int, default=36)
    attack.add_argument("--random-start", action="store_true")
    attack.add_argument("--limit", type=int)
    attack.add_argument("--output-dir", type=Path)
    attack.add_argument("--overwrite", action="store_true")
    attack.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True)
    attack.add_argument("--archive", type=Path)
    attack.set_defaults(handler=command_attack)

    defense = subparsers.add_parser("defense", help="Run dog2 FGSM + JPEG defense")
    add_common(defense)
    defense.add_argument("--model", default="resnet110_cifar10")
    defense.add_argument("--epsilon", type=float, default=8.0)
    defense.add_argument("--compression", type=int, default=70)
    defense.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS / "defense")
    defense.set_defaults(handler=command_defense)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "epsilon") and args.epsilon != 8.0:
        raise ValueError("HW10 requires epsilon to remain fixed at 8 pixels")
    if hasattr(args, "steps") and args.steps < 1:
        raise ValueError("steps must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
