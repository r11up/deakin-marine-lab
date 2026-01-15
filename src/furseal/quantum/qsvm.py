"""QSVC with a ZZFeatureMap quantum kernel, compared against classical SVMs.

Follows the approach studied in
``notebooks/Quantum_Support_Vector_Machines.ipynb`` (Havlicek et al.,
Nature 567, 209-212, 2019), applied to the fur seal data: features are
PCA-reduced to a small number of dimensions (one qubit per dimension),
encoded with a second-order ZZ feature map, and the fidelity quantum
kernel feeds a support vector classifier. Linear and RBF classical SVMs
are trained on the same features as baselines.

The original notebook used the long-deprecated Qiskit Aqua API
(SecondOrderExpansion + QSVM); this module is the qiskit>=1.0 equivalent.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from furseal.quantum.encoding import load_flattened_dataset


def build_qsvc(n_features: int, reps: int = 2, seed: int = 42):
    """ZZFeatureMap (one qubit per feature) -> FidelityQuantumKernel -> QSVC."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed

    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return QSVC(quantum_kernel=kernel)


def reduce_features(X_train: np.ndarray, X_test: np.ndarray, n_components: int,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduce then scale to [0, pi] (angle-friendly range for the
    ZZ feature map's rotation encoding)."""
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_red = pca.fit_transform(X_train)
    X_test_red = pca.transform(X_test)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X_train_red), scaler.transform(X_test_red)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Reduced dataset root (one folder per individual)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--n-features", type=int, default=2,
                        help="PCA components = number of qubits")
    parser.add_argument("--reps", type=int, default=2,
                        help="Feature map depth (repetitions)")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, _ = load_flattened_dataset(args.data, args.image_size,
                                     args.image_size ** 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.seed)
    X_train, X_test = reduce_features(X_train, X_test, args.n_features, args.seed)

    # Classical baselines on the same PCA features.
    linear_svc = svm.LinearSVC(dual="auto").fit(X_train, y_train)
    rbf_svc = svm.SVC(gamma="scale").fit(X_train, y_train)
    print(f"Linear SVM test accuracy: "
          f"{accuracy_score(y_test, linear_svc.predict(X_test)):.4f}")
    print(f"RBF-kernel SVM test accuracy: "
          f"{accuracy_score(y_test, rbf_svc.predict(X_test)):.4f}")

    print("Training QSVC (quantum kernel evaluation is slow)...")
    qsvc = build_qsvc(args.n_features, args.reps, args.seed)
    qsvc.fit(X_train, y_train)
    print(f"QSVC (ZZFeatureMap quantum kernel) test accuracy: "
          f"{accuracy_score(y_test, qsvc.predict(X_test)):.4f}")


if __name__ == "__main__":
    main()
