"""Extract 2048-d ResNet50 embeddings for cropped seal faces.

The ImageNet-pretrained ResNet50 with its classification head removed is
used as a fixed feature extractor. Embeddings feed the clustering stage
(pseudo-labels) and can also serve as input features for the quantum
kernel classifiers.

Ported from ``notebooks/Furl_seal_QML.ipynb``.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

IMAGENET_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    """ResNet50 with the final fully-connected layer removed (outputs 2048-d)."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.to(device).eval()


def extract_embeddings(image_dir: Path, device: torch.device | None = None
                       ) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings [N, 2048], image paths) for all jpgs in a folder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor(device)

    embeddings, image_paths = [], []
    for fname in tqdm(sorted(os.listdir(image_dir)), desc="Extracting embeddings"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = image_dir / fname
        img = Image.open(path).convert("RGB")
        input_tensor = IMAGENET_PREPROCESS(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(input_tensor).squeeze().cpu().numpy()
        embeddings.append(emb)
        image_paths.append(str(path))

    return np.array(embeddings), image_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Folder of cropped face images")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .npy file for the embedding matrix")
    args = parser.parse_args()

    embeddings, image_paths = extract_embeddings(args.images)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    paths_file = args.output.with_suffix(".paths.json")
    with open(paths_file, "w") as f:
        json.dump(image_paths, f, indent=2)

    print(f"Saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} "
          f"to {args.output} (paths: {paths_file})")


if __name__ == "__main__":
    main()
