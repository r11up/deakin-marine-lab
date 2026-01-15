"""Pegasos QSVC with an amplitude-encoding fidelity quantum kernel.

Images are amplitude-encoded (128 pixel amplitudes on 7 qubits via
``RawFeatureVector``), the kernel entry K(x, z) = |<phi(x)|phi(z)>|^2 is
estimated with a state-fidelity computation on a simulator, and the
Pegasos algorithm trains the kernel SVM.

Modern-API port of ``notebooks/pegasos.ipynb`` (which targeted
qiskit 0.44/0.45; the qiskit>=1.0 equivalents are used here).
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from furseal.quantum.encoding import load_flattened_dataset


def build_pegasos_qsvc(n_qubits: int = 7, C: float = 1.0, num_steps: int = 1000,
                       seed: int = 42):
    """Assemble RawFeatureVector -> FidelityQuantumKernel -> PegasosQSVC."""
    from qiskit_machine_learning.algorithms import PegasosQSVC
    from qiskit_machine_learning.circuit.library import RawFeatureVector
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed

    # Amplitude encoding: 2^n_qubits features -> n_qubits qubits.
    feature_map = RawFeatureVector(feature_dimension=2 ** n_qubits)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=num_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Reduced dataset root (one folder per individual), "
                             "e.g. the 128x128 grayscale dataset")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--n-qubits", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Pegasos optimisation steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, label_map = load_flattened_dataset(args.data, args.image_size,
                                             2 ** args.n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.seed)

    pegasos = build_pegasos_qsvc(args.n_qubits, args.C, args.num_steps, args.seed)
    print("Training PegasosQSVC (quantum kernel evaluation is slow; "
          "runtime grows quadratically with the number of samples)...")
    pegasos.fit(X_train, y_train)

    y_pred = pegasos.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nPegasosQSVC accuracy with amplitude encoding: {acc:.4f}\n")

    class_names = sorted(label_map, key=label_map.get)
    present = sorted(set(y_test) | set(y_pred))
    print(classification_report(y_test, y_pred, labels=present,
                                target_names=[class_names[i] for i in present]))


if __name__ == "__main__":
    main()
