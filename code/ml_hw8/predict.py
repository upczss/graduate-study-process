import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import NpyImageDataset
from models import build_model


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CODE_DIR.parent / "data" / "hw8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW8 prediction.csv.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--checkpoint",
        default=str(CODE_DIR / "outputs" / "fcn" / "checkpoints" / "best.pt"),
    )
    parser.add_argument("--output", default=str(CODE_DIR / "prediction.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average original and horizontally flipped reconstruction scores.",
    )
    return parser.parse_args()


def reconstruction_score(
    model,
    images: torch.Tensor,
    use_fp16: bool,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=use_fp16):
        reconstructed = model.reconstruct(images)
    # Root sum of squared pixel errors, matching the original Colab idea.
    return torch.sqrt(
        (reconstructed.float() - images.float()).pow(2).flatten(1).sum(dim=1)
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型：{checkpoint_path}\n请先运行 train.py。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_type"],
        checkpoint["latent_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = NpyImageDataset(
        Path(args.data_dir) / "testingset.npy",
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="Scoring test images"):
            images = images.to(device, non_blocking=True)
            scores = reconstruction_score(model, images, use_fp16)
            if args.tta:
                flipped = torch.flip(images, dims=(3,))
                flipped_scores = reconstruction_score(model, flipped, use_fp16)
                scores = (scores + flipped_scores) / 2.0
            all_scores.append(scores.cpu().numpy())

    scores = np.concatenate(all_scores)
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "score"])
        for image_id, score in enumerate(scores):
            writer.writerow([image_id, f"{float(score):.8f}"])

    print(f"Saved {len(scores)} anomaly scores to {output_path}")
    print(
        f"Score range: min={scores.min():.4f}, "
        f"mean={scores.mean():.4f}, max={scores.max():.4f}"
    )
    if len(scores) != 19636:
        print("Warning: the original HW8 test set should contain 19,636 images.")


if __name__ == "__main__":
    main()
