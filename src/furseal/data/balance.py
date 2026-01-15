"""Balance the raw dataset into a fixed number of images per individual.

The raw dataset is a flat folder where the individual's ID is the first
letter of the filename (a_001.jpg -> individual A). Classes with fewer than
the target number of images are topped up with simple OpenCV augmentations
(flip, rotation, brightness shifts). Output is one folder per individual.

Ported from ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def class_distribution(dataset_path: Path) -> dict[str, int]:
    """Count images per class, where class = first letter of the filename."""
    counts: dict[str, int] = defaultdict(int)
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            counts[filename[0].lower()] += 1
    return dict(sorted(counts.items()))


def augment_image(img):
    """Return a list of simple augmented variants of an image."""
    return [
        cv2.flip(img, 1),                                  # horizontal flip
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.convertScaleAbs(img, alpha=1.2, beta=30),      # brighten
        cv2.convertScaleAbs(img, alpha=0.8, beta=-30),     # darken
    ]


def balance_dataset(src_path: Path, dst_path: Path, target_per_class: int = 10,
                    seed: int | None = 42) -> None:
    """Create a balanced per-class folder structure under ``dst_path``."""
    if seed is not None:
        random.seed(seed)
    os.makedirs(dst_path, exist_ok=True)

    class_images: dict[str, list[str]] = defaultdict(list)
    for fname in os.listdir(src_path):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            class_images[fname[0].lower()].append(fname)

    for cls, images in sorted(class_images.items()):
        class_dir = dst_path / cls.upper()
        os.makedirs(class_dir, exist_ok=True)

        random.shuffle(images)
        src_images = images[:target_per_class]

        for i, fname in enumerate(src_images):
            shutil.copy(src_path / fname, class_dir / f"{cls}_{i + 1}.jpg")

        # Top up under-represented classes with augmented copies.
        if len(src_images) < target_per_class:
            needed = target_per_class - len(src_images)
            for i in range(needed):
                img = cv2.imread(str(src_path / src_images[i % len(src_images)]))
                if img is None:
                    continue
                aug_imgs = augment_image(img)
                aug_img = aug_imgs[i % len(aug_imgs)]
                cv2.imwrite(str(class_dir / f"{cls}_aug_{i + 1}.jpg"), aug_img)

    print(f"Balanced dataset created at: {dst_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Flat folder of raw images (class = filename prefix)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output folder (one subfolder per class)")
    parser.add_argument("--target-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Class distribution before balancing:")
    for cls, count in class_distribution(args.src).items():
        print(f"  Class '{cls.upper()}': {count} images")

    balance_dataset(args.src, args.dst, args.target_per_class, args.seed)


if __name__ == "__main__":
    main()
