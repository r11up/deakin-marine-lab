# Pipeline reference

End-to-end flow from raw field photos to individual identification. Every
stage is a runnable module (`python -m furseal.<stage> --help`); the commands
below assume data lives under `./data` (see `data/README.md`).

## 0. Balance the raw dataset

Raw photos arrive as a flat folder with the individual's ID as the filename
prefix (`a_*.jpg` → seal A). Classes are balanced to a fixed count per
individual, topping up small classes with simple OpenCV augmentations.

```bash
python -m furseal.data.balance \
    --src data/task_datasets --dst data/balanced_task_dataset --target-per-class 10
```

## 1. Face detection (YOLOv5)

A single-class ("face") YOLOv5s detector is fine-tuned from COCO weights on a
manually annotated YOLO-format dataset. Inference with `--save-crop` extracts
face crops.

```bash
python -m furseal.detection.yolo train  --data configs/detection_data.yaml
python -m furseal.detection.yolo detect \
    --weights yolov5/runs/train/exp/weights/best.pt --source data/datasets/test/images
```

Faces can also be cropped directly from the annotation labels:

```bash
python -m furseal.data.crop_faces \
    --images data/datasets/train/images --labels data/datasets/train/labels \
    --output data/cropped_faces
```

## 2. Background removal (SAM)

Segment Anything (ViT-H) with a centre-point prompt isolates the seal; the
best-scoring mask blacks out the background so models learn the animal, not
the beach.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python -m furseal.segmentation.sam_mask \
    --input data/balanced_task_dataset --output data/masked_dataset \
    --checkpoint sam_vit_h_4b8939.pth
```

Optional: binary silhouettes via morphological opening/closing, to test how
much identity signal body shape alone carries.

```bash
python -m furseal.segmentation.morphology \
    --input data/masked_dataset --output data/morp_dataset
```

## 3. Augmentation

Albumentations (flips, rotations, brightness/contrast, shift-scale-rotate)
multiplies the masked dataset before classifier training.

```bash
python -m furseal.data.augment \
    --src data/masked_dataset --dst data/augmented_dataset --count 3
```

## 4. Classical classifier (ResNet50)

ImageNet-pretrained ResNet50 fine-tuned on the augmented masked images
(80/20 split), then evaluated with per-class precision/recall, a confusion
matrix, and one-vs-rest ROC/AUC curves.

```bash
python -m furseal.classical.train \
    --data data/augmented_dataset --output data/models/resnet50_furseal.pth --epochs 10
python -m furseal.classical.evaluate \
    --model data/models/resnet50_furseal.pth --data data/balanced_task_dataset \
    --output-dir eval_outputs
```

## 5. Embeddings + clustering (semi-supervised track)

For unlabelled crops: 2048-d ResNet50 embeddings → PCA(50) → KMeans
pseudo-labels, sanity-checked with t-SNE.

```bash
python -m furseal.features.embeddings \
    --images data/cropped_faces --output data/embeddings/resnet50_embeddings.npy
python -m furseal.features.clustering \
    --embeddings data/embeddings/resnet50_embeddings.npy \
    --output-labels data/embeddings/pseudo_labels.npy --n-clusters 10
```

## 6. Quantum models

Both quantum tracks consume a downscaled grayscale copy of the masked
dataset:

```bash
python -m furseal.data.resize \
    --src data/masked_dataset --dst data/reduced_masked_dataset_128x128 --size 128
```

**Pegasos QSVC, amplitude encoding.** 128 pixel intensities are encoded as
the amplitudes of a 7-qubit state (`RawFeatureVector`); kernel entries
K(x, z) = |⟨φ(x)|φ(z)⟩|² are estimated by state fidelity and the Pegasos
solver trains the kernel SVM.

```bash
python -m furseal.quantum.pegasos --data data/reduced_masked_dataset_128x128
```

**QSVC, ZZ feature map.** Features are PCA-reduced to a few dimensions (one
qubit each), scaled to [0, π], and encoded with a second-order ZZ feature map
(Havlicek et al., Nature 2019). Linear and RBF SVMs on the same features are
reported as classical baselines.

```bash
python -m furseal.quantum.qsvm --data data/reduced_masked_dataset_128x128 --n-features 2
```

Both run on the local Qiskit primitive simulators by default. Kernel
evaluation cost grows quadratically with sample count — keep the datasets
small (tens of images per class).
