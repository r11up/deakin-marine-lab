# Notebooks

Original research/exploration notebooks this repository was built from. The
production code in `src/furseal/` is the cleaned-up, scriptable port of these;
paths inside the notebooks still reference the original Google Drive layout
(`/content/drive/MyDrive/Capstone/Deakin/furseal/...`).

| Notebook | Contents | Ported to |
|---|---|---|
| `furseal_face_detection.ipynb` | YOLOv5 seal face detector: training and inference with cropped outputs | `furseal.detection.yolo` |
| `Furl_seal_QML.ipynb` | Face cropping from YOLO labels, ResNet50 embeddings, PCA + KMeans pseudo-labels, t-SNE | `furseal.data.crop_faces`, `furseal.features.*` |
| `sam_fur_seal.ipynb` | Dataset balancing, SAM background masking, albumentations augmentation, ResNet50 classifier + evaluation, morphological masks | `furseal.data.balance`, `furseal.data.augment`, `furseal.segmentation.*`, `furseal.classical.*` |
| `pegasos.ipynb` | Image downscaling, amplitude encoding (7 qubits / 128 dims), Pegasos QSVC with fidelity quantum kernel | `furseal.data.resize`, `furseal.quantum.encoding`, `furseal.quantum.pegasos` |
| `Quantum_Support_Vector_Machines.ipynb` | QSVM theory + tutorial (breast cancer dataset, legacy Qiskit Aqua API) | `furseal.quantum.qsvm` (modern API, applied to seal data) |
