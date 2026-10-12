"""Optional baseline visualization: time-resolved PCA of trial-average firing rates.

This is NOT LFADS and not a single-trial latent dynamical model.
Usage: python scripts/06_pca_preview.py <path-to-h5>
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--zscore-neurons", action="store_true")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(args.h5_path, "r") as f:
        train = f["train_encod_data"][:]
        valid = f["valid_encod_data"][:]
        edges = f["bin_edges_s"][:] if "bin_edges_s" in f else None
        bin_size = float(f.attrs["bin_size_s"]) if "bin_size_s" in f.attrs else 0.01

    counts = np.concatenate((train, valid), axis=0)
    if counts.shape[2] < 2:
        raise ValueError("Need at least two neurons for PCA preview")
    # Each row is a neural population vector at a time bin, averaged over trials.
    rates_hz = counts.mean(axis=0) / bin_size
    centered = rates_hz - rates_hz.mean(axis=0, keepdims=True)
    if args.zscore_neurons:
        std = centered.std(axis=0, ddof=0)
        centered = centered / np.where(std > 0, std, 1)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(3, centered.shape[0], centered.shape[1])
    scores = u[:, :n_components] * singular_values[:n_components]
    explained = singular_values ** 2 / max(1, np.sum(singular_values ** 2))
    centers = (edges[:-1] + edges[1:]) / 2 if edges is not None else np.arange(len(scores)) * bin_size

    fig = plt.figure(figsize=(9, 6))
    if n_components >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(scores[:, 0], scores[:, 1], scores[:, 2], alpha=0.5)
        scatter = ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=centers, cmap="viridis", s=16)
        ax.set_zlabel("PC3")
    else:
        ax = fig.add_subplot(111)
        ax.plot(scores[:, 0], scores[:, 1], alpha=0.5)
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=centers, cmap="viridis", s=16)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trial-averaged firing-rate PCA (not LFADS)")
    fig.colorbar(scatter, ax=ax, label="Time relative to event (s)")
    fig.tight_layout()
    out = args.output or args.h5_path.with_name(args.h5_path.stem + "_pca.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("PCA explained variance:", np.round(explained[:n_components], 3))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
