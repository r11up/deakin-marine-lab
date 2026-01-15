"""Augment the masked dataset with Albumentations.

For every image, the original is copied through and ``count`` augmented
variants are generated (flips, small rotations, brightness/contrast and
shift-scale-rotate jitter). This multiplies the effective dataset size for
classifier training.

Ported from ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
    ])


def augment_and_save(image_rgb, output_path_base: str | Path, count: int = 3,
                     transform: A.Compose | None = None) -> None:
    """Write ``count`` augmented variants of an RGB image to disk."""
    transform = transform or build_transform()
    for i in range(count):
        aug_img = transform(image=image_rgb)["image"]
        aug_path = f"{output_path_base}_aug_{i + 1}.jpg"
        cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))


def augment_dataset(src_root: Path, dst_root: Path, count: int = 3) -> None:
    """Copy originals and add augmented variants, class folder by class folder."""
    os.makedirs(dst_root, exist_ok=True)
    transform = build_transform()

    for class_name in sorted(os.listdir(src_root)):
        input_dir = src_root / class_name
        if not input_dir.is_dir() or class_name.startswith("."):
            continue
        output_dir = dst_root / class_name
        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(sorted(os.listdir(input_dir)),
                             desc=f"Augmenting {class_name}"):
            if not filename.lower().endswith(IMAGE_EXTENSIONS) or filename.startswith("."):
                continue
            image = cv2.imread(str(input_dir / filename))
            if image is None:
                print(f"Failed to load: {input_dir / filename}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(str(output_dir / filename), image)

            base_filename = os.path.splitext(filename)[0]
            augment_and_save(image_rgb, output_dir / base_filename, count, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Masked dataset root (one folder per class)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Augmented dataset root")
    parser.add_argument("--count", type=int, default=3,
                        help="Augmented variants per image")
    args = parser.parse_args()

    augment_dataset(args.src, args.dst, args.count)


if __name__ == "__main__":
    main()
