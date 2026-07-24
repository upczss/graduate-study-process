"""李宏毅机器学习 HW3：Food-11 图像分类。

示例：
  # 训练，并在验证集表现最好时保存模型
  python 1.py --mode train --epochs 30

  # 使用最佳模型预测测试集，生成 prediction.csv
  python 1.py --mode predict

  # 训练结束后立即预测
  python 1.py --mode all --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data" / "hw3"
NUM_CLASSES = 11


def set_seed(seed: int) -> None:
    """尽可能固定随机性，方便复现实验结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    """选择计算设备，并识别“CUDA 可见但安装包不支持显卡架构”的情况。"""
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 检测不到可用 GPU")
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability()
    required_arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    if required_arch not in supported_arches:
        message = (
            f"当前 PyTorch 不支持显卡架构 {required_arch} "
            f"（安装包支持：{', '.join(supported_arches)}）。"
            "请安装支持该架构的 PyTorch CUDA 版本。"
        )
        if requested == "cuda":
            raise RuntimeError(message)
        warnings.warn(message + " 本次自动改用 CPU。", stacklevel=2)
        return torch.device("cpu")
    return torch.device("cuda")


class FoodDataset(Dataset):
    """读取 Food-11 图片；训练/验证标签来自文件名开头。"""

    def __init__(self, folder: Path, transform, is_test: bool = False) -> None:
        self.folder = Path(folder)
        self.transform = transform
        self.is_test = is_test
        self.files = sorted(
            self.folder.glob("*.jpg"),
            key=lambda path: (
                int(path.stem) if is_test else int(path.stem.split("_", 1)[1])
            ),
        )
        if not self.files:
            raise FileNotFoundError(f"在 {self.folder} 中没有找到 jpg 图片")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        # 用 with 保证大量读取图片时及时关闭文件句柄。
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        if self.is_test:
            return image, path.stem

        try:
            label = int(path.stem.split("_", 1)[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"无法从文件名解析标签：{path.name}") from exc
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"标签应在 0~{NUM_CLASSES - 1}，实际为 {label}")
        return image, label


class CNNBlock(nn.Module):
    """一个卷积层加 Batch Normalization，不在块内执行激活。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualNetwork(nn.Module):
    """用户原先设计的六个主卷积层残差 CNN（修正版）。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.cnn_layer1 = CNNBlock(3, 64, 3, 1, 1)
        self.cnn_layer2 = CNNBlock(64, 64, 3, 1, 1)
        self.cnn_layer3 = CNNBlock(64, 128, 3, 2, 1)
        self.cnn_layer4 = CNNBlock(128, 128, 3, 1, 1)
        self.cnn_layer5 = CNNBlock(128, 256, 3, 2, 1)
        self.cnn_layer6 = CNNBlock(256, 256, 3, 1, 1)

        # layer3 和 layer5 会同时减半宽高、增加通道数。
        # 原代码直接相加会尺寸不匹配，因此用 1×1 卷积投影残差。
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        # 保留 256×32×32 的超大全连接层会产生约 6700 万个参数，
        # 很容易占满显存。全局平均池化能保留 256 个通道信息且更稳。
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一组：输入通道数和尺寸不变，可直接建立残差连接。
        x = self.relu(self.cnn_layer1(x))
        residual = x
        x = self.cnn_layer2(x)
        x = self.relu(x + residual)

        # 第二组：用 shortcut2 将残差变为 128×64×64。
        residual = self.shortcut2(x)
        x = self.relu(self.cnn_layer3(x))
        x = self.cnn_layer4(x)
        x = self.relu(x + residual)

        # 第三组：用 shortcut3 将残差变为 256×32×32。
        residual = self.shortcut3(x)
        x = self.relu(self.cnn_layer5(x))
        x = self.cnn_layer6(x)
        x = self.relu(x + residual)

        return self.fc_layer(self.pool(x))


# 保留训练和预测代码使用的统一名称。
Classifier = ResidualNetwork


def build_transforms(image_size: int):
    # 训练集才使用随机增强；验证集和测试集必须保持确定性。
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return train_transform, eval_transform


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    description = "train" if is_training else "valid"
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train(args: argparse.Namespace, device: torch.device) -> None:
    train_transform, eval_transform = build_transforms(args.image_size)
    train_set = FoodDataset(args.data_dir / "training", train_transform)
    valid_set = FoodDataset(args.data_dir / "validation", eval_transform)
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers, device
    )
    valid_loader = make_loader(
        valid_set, args.batch_size, False, args.num_workers, device
    )

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy = -1.0
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} | train={len(train_set)} | valid={len(valid_set)} "
        f"| parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        with torch.inference_mode():
            valid_loss, valid_accuracy = run_epoch(
                model, valid_loader, criterion, device
            )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"valid loss {valid_loss:.4f}, acc {valid_accuracy:.4f}"
        )

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "valid_accuracy": valid_accuracy,
                    "epoch": epoch,
                    "image_size": args.image_size,
                },
                args.model_path,
            )
            print(f"  已保存新的最佳模型：{args.model_path}")

    print(f"训练完成，最佳验证准确率：{best_accuracy:.4f}")


def predict(args: argparse.Namespace, device: torch.device) -> None:
    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"找不到模型 {args.model_path}，请先使用 --mode train 训练"
        )

    checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    image_size = int(checkpoint.get("image_size", args.image_size))
    _, eval_transform = build_transforms(image_size)
    test_set = FoodDataset(
        args.data_dir / "test", eval_transform, is_test=True
    )
    test_loader = make_loader(
        test_set, args.batch_size, False, args.num_workers, device
    )

    model = Classifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows: list[tuple[str, int]] = []

    with torch.inference_mode():
        for images, image_ids in tqdm(test_loader, desc="predict"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                predictions = model(images).argmax(dim=1).cpu().tolist()
            rows.extend(zip(image_ids, predictions))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(rows)
    print(f"已生成 {args.output_path}，共 {len(rows)} 条预测")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food-11 image classification")
    parser.add_argument(
        "--mode", choices=("train", "predict", "all"), default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--model-path", type=Path, default=SCRIPT_DIR / "best_model.pt"
    )
    parser.add_argument(
        "--output-path", type=Path, default=SCRIPT_DIR / "prediction.csv"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto 会优先使用兼容的 CUDA，否则使用 CPU",
    )
    # Windows 下默认 0 最稳；显存/内存充足时可手动设成 2~4。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5201314)
    args = parser.parse_args()

    args.data_dir = args.data_dir.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs、batch-size 必须为正，image-size 至少为 32")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    if args.mode in ("train", "all"):
        train(args, device)
    if args.mode in ("predict", "all"):
        predict(args, device)


if __name__ == "__main__":
    main()
