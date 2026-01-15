"""Resize a per-class dataset to small grayscale images.

Used to produce the reduced datasets consumed by the quantum models
(e.g. 128x128 for amplitude encoding on 7 qubits, or tiny 4x4 / 5x5
variants for experimentation).

Ported from ``notebooks/pegasos.ipynb``.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def preprocess_image(img_path: str | Path, size: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Load an image as a flat, [0, 1]-normalised grayscale feature vector."""
    img = Image.open(img_path).convert("L").resize(size)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array.flatten()


def resize_dataset(src_root: Path, dst_root: Path, size: tuple[int, int]) -> None:
    """Resize every image in ``src_root/<CLASS>/`` to grayscale ``size``."""
    os.makedirs(dst_root, exist_ok=True)

    class_folders = sorted(d for d in os.listdir(src_root)
                           if (src_root / d).is_dir() and not d.startswith("."))
    for folder in class_folders:
        dst_folder = dst_root / folder
        os.makedirs(dst_folder, exist_ok=True)

        for filename in sorted(os.listdir(src_root / folder)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            try:
                img = Image.open(src_root / folder / filename).convert("L").resize(size)
                img.save(dst_folder / filename)
            except Exception as e:
                print(f"Failed to process {folder}/{filename}: {e}")

    print(f"Resized dataset saved at: {dst_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Source dataset root (one folder per class)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output dataset root")
    parser.add_argument("--size", type=int, default=128,
                        help="Target side length in pixels (images become size x size)")
    args = parser.parse_args()

    resize_dataset(args.src, args.dst, (args.size, args.size))


if __name__ == "__main__":
    main()
