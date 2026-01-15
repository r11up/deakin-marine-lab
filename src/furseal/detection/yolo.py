"""YOLOv5 seal face detection: training and inference wrappers.

Thin wrappers around the ultralytics/yolov5 repository scripts, mirroring
the commands used in ``notebooks/furseal_face_detection.ipynb``:

    python train.py  --img 416 --batch 16 --epochs 150 \\
                     --data <data.yaml> --weights yolov5s.pt --cache
    python detect.py --weights <best.pt> --img 416 --conf 0.1 \\
                     --source <images> --save-crop

The YOLOv5 repo is cloned on first use. The detection dataset must be in
YOLO format with a ``data.yaml`` (see configs/detection_data.yaml for a
template).
"""

import argparse
import subprocess
import sys
from pathlib import Path

YOLOV5_REPO_URL = "https://github.com/ultralytics/yolov5"


def ensure_yolov5(repo_dir: Path) -> Path:
    """Clone the YOLOv5 repository if it is not already present."""
    if not (repo_dir / "train.py").exists():
        print(f"Cloning YOLOv5 into {repo_dir} ...")
        subprocess.run(["git", "clone", YOLOV5_REPO_URL, str(repo_dir)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(repo_dir / "requirements.txt")], check=True)
    return repo_dir


def train(data_yaml: Path, repo_dir: Path, img_size: int = 416, batch: int = 16,
          epochs: int = 150, weights: str = "yolov5s.pt") -> None:
    """Train a custom seal face detector, starting from COCO-pretrained weights."""
    ensure_yolov5(repo_dir)
    subprocess.run([
        sys.executable, "train.py",
        "--img", str(img_size),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", str(Path(data_yaml).resolve()),
        "--weights", weights,
        "--cache",
    ], cwd=repo_dir, check=True)


def detect(weights: Path, source: Path, repo_dir: Path, img_size: int = 416,
           conf: float = 0.1, save_crop: bool = True) -> None:
    """Run inference; ``save_crop`` writes cropped face images alongside results."""
    ensure_yolov5(repo_dir)
    cmd = [
        sys.executable, "detect.py",
        "--weights", str(Path(weights).resolve()),
        "--img", str(img_size),
        "--conf", str(conf),
        "--source", str(Path(source).resolve()),
    ]
    if save_crop:
        cmd.append("--save-crop")
    subprocess.run(cmd, cwd=repo_dir, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train the face detector")
    p_train.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    p_train.add_argument("--repo-dir", type=Path, default=Path("yolov5"))
    p_train.add_argument("--img-size", type=int, default=416)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--epochs", type=int, default=150)
    p_train.add_argument("--weights", default="yolov5s.pt")

    p_detect = subparsers.add_parser("detect", help="Run face detection")
    p_detect.add_argument("--weights", type=Path, required=True,
                          help="Trained weights, e.g. yolov5/runs/train/exp/weights/best.pt")
    p_detect.add_argument("--source", type=Path, required=True, help="Image folder")
    p_detect.add_argument("--repo-dir", type=Path, default=Path("yolov5"))
    p_detect.add_argument("--img-size", type=int, default=416)
    p_detect.add_argument("--conf", type=float, default=0.1)
    p_detect.add_argument("--no-save-crop", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.repo_dir, args.img_size, args.batch,
              args.epochs, args.weights)
    else:
        detect(args.weights, args.source, args.repo_dir, args.img_size,
               args.conf, save_crop=not args.no_save_crop)


if __name__ == "__main__":
    main()
