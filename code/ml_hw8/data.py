from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyImageDataset(Dataset):
    """Read HW8 images lazily from a memory-mapped .npy file."""

    def __init__(
        self,
        path: str | Path,
        indices: list[int] | np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.path = str(path)
        self.images = np.load(self.path, mmap_mode="r")
        if self.images.ndim != 4 or self.images.shape[1:] != (64, 64, 3):
            raise ValueError(
                f"Expected (#images, 64, 64, 3), got {self.images.shape}"
            )
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(self.images), dtype=np.int64)
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        # copy() avoids creating a tensor backed by a read-only memmap.
        image = self.images[self.indices[index]].copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
        return image


def split_indices(
    total: int,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(total)
    validation_size = max(1, int(total * validation_ratio))
    return indices[validation_size:], indices[:validation_size]

