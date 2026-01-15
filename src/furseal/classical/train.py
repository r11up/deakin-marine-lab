"""Train a ResNet50 classifier to identify individual fur seals.

Fine-tunes an ImageNet-pretrained ResNet50 on a per-class image folder
(one folder per individual: A/, B/, ... H/), with an 80/20 train/val
split. Works on the augmented masked dataset as well as the morphological
silhouette dataset (pass ``--no-normalize`` for the latter, matching the
original experiments).

Ported from ``notebooks/sam_fur_seal.ipynb``.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(normalize: bool = True) -> transforms.Compose:
    steps = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_model(num_classes: int, device: torch.device,
                pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def make_loaders(data_dir: Path, batch_size: int = 16, val_fraction: float = 0.2,
                 normalize: bool = True, seed: int = 42):
    """Return (train_loader, val_loader, class_names)."""
    dataset = ImageFolder(str(data_dir), transform=build_transform(normalize))
    val_size = int(val_fraction * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset.classes


def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device,
                num_epochs: int = 10, lr: float = 1e-4) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = len(train_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = 100 * correct / n_train
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {running_loss / n_train:.4f}, Accuracy: {train_acc:.2f}%")

    return model


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
    return 100 * correct / len(val_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset root (one folder per individual)")
    parser.add_argument("--output", type=Path, default=Path("resnet50_furseal.pth"),
                        help="Where to save the trained weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip ImageNet normalisation (used for the "
                             "morphological silhouette dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = make_loaders(
        args.data, args.batch_size, args.val_fraction,
        normalize=not args.no_normalize, seed=args.seed)
    print("Classes:", class_names)

    model = build_model(len(class_names), device)
    model = train_model(model, train_loader, device, args.epochs, args.lr)

    val_acc = validate(model, val_loader, device)
    print(f"Validation Accuracy: {val_acc:.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved at {args.output}")


if __name__ == "__main__":
    main()
