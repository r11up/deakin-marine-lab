"""Morphological mask extraction from SAM-masked images.

Converts each masked image to a clean binary silhouette: grayscale ->
binary threshold -> morphological opening (erosion + dilation, removes
speckle noise) -> closing (dilation + erosion, fills small holes). The
resulting silhouette dataset is used to test whether body shape alone
carries identity information.

Ported from ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def default_kernel() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))


def morphological_mask(image: np.ndarray, kernel: np.ndarray | None = None,
                       thresh: int = 50, maxval: int = 100) -> np.ndarray:
    """Compute the opened-then-closed binary mask of a BGR image."""
    kernel = kernel if kernel is not None else default_kernel()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def process_dataset(input_root: Path, output_root: Path) -> None:
    """Write a morphological mask for every image in a per-class dataset."""
    kernel = default_kernel()

    for subdir in sorted(os.listdir(input_root)):
        input_folder = input_root / subdir
        if not input_folder.is_dir() or subdir.startswith("."):
            continue
        output_folder = output_root / subdir
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing {subdir}:")
        for img_file in tqdm(sorted(os.listdir(input_folder))):
            image = cv2.imread(str(input_folder / img_file))
            if image is None:
                print(f"Image not found: {input_folder / img_file}")
                continue
            mask = morphological_mask(image, kernel)
            cv2.imwrite(str(output_folder / img_file), mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Masked dataset root (one folder per class)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Morphological mask dataset root")
    args = parser.parse_args()

    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
