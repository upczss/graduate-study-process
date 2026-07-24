from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data" / "hw8"


def check_file(filename: str, expected_count: int) -> None:
    path = DATA_DIR / filename
    images = np.load(path, mmap_mode="r")
    print(
        f"{filename}: shape={images.shape}, dtype={images.dtype}, "
        f"range=[{images[:100].min()}, {images[:100].max()}]"
    )
    if images.shape != (expected_count, 64, 64, 3):
        raise ValueError(f"Unexpected shape for {filename}: {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"Unexpected dtype for {filename}: {images.dtype}")


def main() -> None:
    check_file("trainingset.npy", 100000)
    check_file("testingset.npy", 19636)
    print("HW8 data check passed.")


if __name__ == "__main__":
    main()

