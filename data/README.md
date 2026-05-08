# Data directory

Datasets and trained models are **not** versioned in this repository (see
`.gitignore`). Place the project data here — or anywhere else and point the
scripts at it with `FURSEAL_DATA_ROOT` / the per-script path arguments.

Expected layout (produced/consumed by the pipeline stages):

```
data/
├── task_datasets/                 # raw photos, class in filename prefix (a_*.jpg ... h_*.jpg)
├── balanced_task_dataset/         # furseal.data.balance      → A/ B/ ... H/
├── masked_dataset/                # furseal.segmentation.sam_mask
├── augmented_dataset/             # furseal.data.augment
├── morp_dataset/                  # furseal.segmentation.morphology
├── reduced_masked_dataset_128x128/# furseal.data.resize (quantum input)
├── datasets/                      # YOLO detection dataset (train/valid/test + data.yaml)
├── cropped_faces/                 # furseal.data.crop_faces
├── embeddings/                    # furseal.features.embeddings (.npy)
└── models/                        # trained weights (.pth)
```
