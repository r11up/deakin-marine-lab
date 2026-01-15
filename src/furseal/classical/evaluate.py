"""Evaluate a trained ResNet50 seal classifier on a per-class dataset.

Produces overall accuracy, a per-class classification report, a confusion
matrix heatmap, and one-vs-rest multiclass ROC curves with AUC — the same
diagnostics used in ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import label_binarize
from torchvision import models
from tqdm import tqdm

from furseal.classical.train import build_transform


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_dataset(model: nn.Module, data_dir: Path, class_names: list[str],
                    device: torch.device, normalize: bool = True):
    """Run the model over every image; return (y_true, y_pred, y_score)."""
    transform = build_transform(normalize)
    y_true, y_pred, y_score = [], [], []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        image_files = [f for f in sorted(os.listdir(class_dir))
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
            image = Image.open(class_dir / image_file).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)

            y_true.append(class_idx)
            y_pred.append(int(output.argmax(1).item()))
            y_score.append(probs.cpu().numpy()[0])

    return np.array(y_true), np.array(y_pred), np.vstack(y_score)


def plot_confusion_matrix(y_true, y_pred, class_names: list[str],
                          output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_score, class_names: list[str],
                    output_path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(12, 8))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Trained weights (.pth)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Evaluation dataset root (one folder per individual)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = sorted(d for d in os.listdir(args.data)
                         if (args.data / d).is_dir() and not d.startswith("."))

    model = load_model(args.model, len(class_names), device)
    y_true, y_pred, y_score = predict_dataset(
        model, args.data, class_names, device, normalize=not args.no_normalize)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names,
                          args.output_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_score, class_names,
                    args.output_dir / "roc_curves.png")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
