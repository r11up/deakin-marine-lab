"""Cluster face embeddings into pseudo-labels and visualise with t-SNE.

PCA reduces the 2048-d ResNet50 embeddings to 50 components, KMeans
assigns cluster IDs (pseudo-labels for the unlabelled crops), and t-SNE
projects the reduced embeddings to 2-D for a visual check of cluster
separation.

Ported from ``notebooks/Furl_seal_QML.ipynb``.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 10,
                       n_components: int = 50, random_state: int = 42
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster labels, PCA-reduced embeddings)."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(reduced)
    return labels, reduced


def plot_tsne(reduced_embeddings: np.ndarray, cluster_labels: np.ndarray,
              output_path: Path, perplexity: float = 30,
              random_state: int = 42) -> None:
    """Save a 2-D t-SNE scatter coloured by cluster ID."""
    perplexity = min(perplexity, max(len(reduced_embeddings) - 1, 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(reduced_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap="tab10")
    plt.title("t-SNE Clustering of Fur Seal Faces")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster ID")
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help=".npy file produced by furseal.features.embeddings")
    parser.add_argument("--output-labels", type=Path, required=True,
                        help="Output .npy file for the pseudo-labels")
    parser.add_argument("--output-plot", type=Path, default=Path("tsne_clusters.png"))
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    labels, reduced = cluster_embeddings(embeddings, args.n_clusters,
                                         args.pca_components, args.seed)

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_labels, labels)
    plot_tsne(reduced, labels, args.output_plot, random_state=args.seed)

    print(f"Pseudo-labels saved to {args.output_labels}")
    print(f"t-SNE plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
