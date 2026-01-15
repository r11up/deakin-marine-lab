"""Background removal with Meta's Segment Anything Model (SAM).

For every image, a single positive point prompt at the image centre selects
the seal; the highest-scoring predicted mask is applied so that only the
animal remains and the background is blacked out. Masked images are what
the classifiers train on, which forces them to learn from the seal itself
rather than the (highly location-correlated) background.

Requires the ``segment-anything`` package and a SAM checkpoint, e.g.:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Ported from ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_predictor(checkpoint: Path, model_type: str = "vit_h", device: str | None = None):
    """Load a SAM checkpoint and return a SamPredictor."""
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}. Download it with:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def predict_center_mask(predictor, image_rgb: np.ndarray) -> np.ndarray:
    """Predict masks from a centre-point prompt and return the best-scoring one."""
    predictor.set_image(image_rgb)
    input_point = np.array([[image_rgb.shape[1] // 2, image_rgb.shape[0] // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def apply_mask_and_save(image_rgb: np.ndarray, mask: np.ndarray,
                        output_path: str | Path) -> None:
    """Black out everything outside the mask and save as BGR jpeg."""
    binary_mask = mask.astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
    cv2.imwrite(str(output_path), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))


def mask_dataset(input_root: Path, output_root: Path, predictor) -> None:
    """Apply SAM centre-point masking to every image, class folder by class folder."""
    os.makedirs(output_root, exist_ok=True)

    class_folders = sorted(d for d in os.listdir(input_root) if not d.startswith("."))
    for class_name in class_folders:
        class_input_dir = input_root / class_name
        if not class_input_dir.is_dir():
            continue
        class_output_dir = output_root / class_name
        os.makedirs(class_output_dir, exist_ok=True)

        for filename in tqdm(sorted(os.listdir(class_input_dir)),
                             desc=f"Processing {class_name}"):
            if not filename.lower().endswith(IMAGE_EXTENSIONS) or filename.startswith("."):
                continue
            image = cv2.imread(str(class_input_dir / filename))
            if image is None:
                print(f"Failed to read: {class_input_dir / filename}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            best_mask = predict_center_mask(predictor, image_rgb)
            apply_mask_and_save(image_rgb, best_mask, class_output_dir / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Balanced dataset root (one folder per class)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Masked dataset root")
    parser.add_argument("--checkpoint", type=Path, default=Path("sam_vit_h_4b8939.pth"))
    parser.add_argument("--model-type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    args = parser.parse_args()

    predictor = load_predictor(args.checkpoint, args.model_type, args.device)
    mask_dataset(args.input, args.output, predictor)


if __name__ == "__main__":
    main()
