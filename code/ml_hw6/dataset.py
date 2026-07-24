from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_image_root(data_dir: Path) -> tuple[Path, bool]:
    """
    Return (root, needs_synthetic_class).

    ImageFolder requires a class subdirectory. The original HW6 dataset is
    usually a flat `faces/*.jpg` directory, so that case is handled separately.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    direct_images = [
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return data_dir, bool(direct_images)


class FlatImageFolder(datasets.VisionDataset):
    """A label-free dataset for a directory containing image files."""

    def __init__(self, root: Path, transform=None) -> None:
        super().__init__(str(root), transform=transform)
        self.samples = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise RuntimeError(f"No supported images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = datasets.folder.default_loader(str(self.samples[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def build_dataloader(
    data_dir: str,
    batch_size: int,
    image_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    root, is_flat = _find_image_root(Path(data_dir))
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )

    dataset = (
        FlatImageFolder(root, transform=transform)
        if is_flat
        else datasets.ImageFolder(str(root), transform=transform)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

