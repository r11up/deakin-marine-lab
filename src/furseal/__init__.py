"""Fur seal face recognition — Deakin University / Tulip Lab capstone project.

Pipeline stages:
    data          - dataset balancing, cropping, resizing, augmentation
    detection     - YOLOv5 seal face detection (train / inference wrappers)
    segmentation  - SAM background masking and morphological masking
    features      - ResNet50 embeddings, clustering for pseudo-labels
    classical     - ResNet50 individual classifier (train / evaluate)
    quantum       - quantum kernel methods (Pegasos QSVC, QSVC)
"""

__version__ = "0.1.0"
