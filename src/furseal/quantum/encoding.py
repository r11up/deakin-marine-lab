"""Classical-to-quantum data encoding for the fur seal dataset.

Images are downscaled to grayscale, flattened, and L2-normalised so the
pixel vector can be amplitude-encoded into a quantum state: a 128-dim
unit vector maps onto the amplitudes of a 7-qubit state (2^7 = 128).

Ported from ``notebooks/pegasos.ipynb``.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def amplitude_encode(vec: np.ndarray, max_dim: int) -> np.ndarray:
    """Trim to ``max_dim`` features and L2-normalise (amplitude encoding
    requires a unit-norm state vector)."""
    vec = np.asarray(vec, dtype=np.float32).flatten()[:max_dim]
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot amplitude-encode an all-zero vector")
    return vec / norm


def load_flattened_dataset(data_path: Path, image_size: int = 128,
                           max_dim: int = 2 ** 7
                           ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load a per-class image dataset as amplitude-encoded feature vectors.

    Returns (X [N, max_dim], y [N], label_map {class name -> label index}).
    """
    class_folders = sorted(d for d in os.listdir(data_path)
                           if (data_path / d).is_dir() and not d.startswith("."))
    label_map = {folder: idx for idx, folder in enumerate(class_folders)}

    X, y = [], []
    for folder, label in label_map.items():
        folder_path = data_path / folder
        for file in sorted(os.listdir(folder_path)):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            img = Image.open(folder_path / file).convert("L") \
                       .resize((image_size, image_size))
            vec = amplitude_encode(np.array(img), max_dim)
            X.append(vec)
            y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"Loaded {X.shape[0]} samples of dimension {X.shape[1]}, "
          f"labels: {label_map}")
    return X, y, label_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset root (one folder per individual)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--n-qubits", type=int, default=7,
                        help="Feature dimension is 2^n_qubits")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional .npz output (X, y)")
    args = parser.parse_args()

    X, y, label_map = load_flattened_dataset(args.data, args.image_size,
                                             2 ** args.n_qubits)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, X=X, y=y,
                 classes=np.array(sorted(label_map, key=label_map.get)))
        print(f"Encoded dataset saved to {args.output}")


if __name__ == "__main__":
    main()
