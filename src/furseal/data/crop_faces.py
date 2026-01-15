"""Crop seal faces out of full images using YOLO-format label files.

Each label file contains one line per face: ``class x_center y_center w h``
with coordinates normalised to [0, 1]. Crops are written as
``<image-stem>_<box-index>.jpg``.

Ported from ``notebooks/Furl_seal_QML.ipynb``.
"""

import argparse
import os
from pathlib import Path

import cv2


def crop_faces(image_dir: Path, label_dir: Path, output_dir: Path) -> int:
    """Crop every labelled bounding box; return the number of crops written."""
    os.makedirs(output_dir, exist_ok=True)
    n_crops = 0

    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith(".txt"):
            continue

        img_name = label_file.replace(".txt", ".jpg")
        image = cv2.imread(str(image_dir / img_name))
        if image is None:
            continue
        height, width = image.shape[:2]

        with open(label_dir / label_file) as f:
            for idx, line in enumerate(f.readlines()):
                _, x_center, y_center, w, h = map(float, line.strip().split())
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                cropped_face = image[max(y1, 0):y2, max(x1, 0):x2]
                if cropped_face.size == 0:
                    continue
                out_path = output_dir / f"{img_name[:-4]}_{idx}.jpg"
                cv2.imwrite(str(out_path), cropped_face)
                n_crops += 1

    return n_crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Folder with source images")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Folder with YOLO-format .txt label files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Folder for cropped face images")
    args = parser.parse_args()

    n = crop_faces(args.images, args.labels, args.output)
    print(f"Cropping complete: {n} faces written to {args.output}")


if __name__ == "__main__":
    main()
