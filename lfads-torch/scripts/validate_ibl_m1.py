#!/usr/bin/env python3
"""Validate an existing single-session lfads-torch run without retraining.

Usage:
  python scripts/validate_ibl_m1.py
  python scripts/validate_ibl_m1.py --run-dir /path/to/run
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # Save figures on a remote/headless server.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DEFAULT_INPUT_H5 = Path("/home/changhyun/LFADS_pra/lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5")

DEFAULT_RUN_DIR = Path(
    "/home/changhyun/LFADS_pra/lfads-torch/runs/"
    "lfads-torch-example/ibl_m1/261012_162141_ibl_m1"
)
EPS = 1e-10


def log_metric(csv_df, name):
    """Extract epoch-level logged values while ignoring sparse CSV rows."""
    if "epoch" not in csv_df or name not in csv_df:
        return None
    tmp = csv_df[["epoch", name]].dropna()
    if tmp.empty:
        return None
    return tmp.groupby("epoch", sort=True)[name].last()


def poisson_bps(y, lam, baseline):
    """Log-likelihood gain over a constant per-neuron train baseline, bits/spike.

    y, lam shape: (validation trials, bins, neurons)
    baseline shape: (1, 1, neurons)
    """
    lam = np.clip(lam, EPS, None)
    base = np.clip(baseline, EPS, None)
    log_gain = y * (np.log(lam) - np.log(base)) - (lam - base)
    counts = y.sum(axis=(0, 1))
    gain_by_unit = log_gain.sum(axis=(0, 1))
    bps_by_unit = np.divide(
        gain_by_unit,
        counts * np.log(2),
        out=np.full(len(counts), np.nan, dtype=float),
        where=counts > 0,
    )
    total_spikes = counts.sum()
    total_bps = float(log_gain.sum() / (total_spikes * np.log(2))) if total_spikes > 0 else float("nan")
    return total_bps, bps_by_unit, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--sigma-bins", type=float, default=3.0,
                        help="Gaussian smoothing for visualization only (bins)")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"LFADS run folder not found: {run_dir}")

    outputs = sorted(run_dir.glob("lfads_output_*.h5"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one LFADS output HDF5, found: {outputs}")
    output_h5 = outputs[0]
    metrics_path = run_dir / "csv_logs" / "metrics.csv"

    # A. Learning curves, if metrics.csv exists.
    if metrics_path.is_file():
        df = pd.read_csv(metrics_path)
        print("Logged metrics:", [k for k in df.columns if "recon" in k or "loss" in k])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        metric_groups = [
            (axes[0], ["train/loss", "valid/loss"], "Total loss (with regularizers)"),
            (axes[1], ["train/recon", "valid/recon", "valid/recon_smth"], "Poisson reconstruction loss"),
        ]
        for ax, names, title in metric_groups:
            for name in names:
                series = log_metric(df, name)
                if series is not None:
                    ax.plot(series.index.to_numpy() + 1, series.to_numpy(), marker="o", label=name)
                    print(f"{name}: first={series.iloc[0]:.5g}, last={series.iloc[-1]:.5g}")
            ax.set_xlabel("Epoch (1-based)")
            ax.set_ylabel("Logged value")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.tight_layout()
        curve_path = run_dir / "validation_learning_curves.png"
        fig.savefig(curve_path, dpi=180)
        plt.close(fig)
        print("Saved:", curve_path)
    else:
        print("No metrics CSV found at:", metrics_path)

    # B. Get spike counts and bin time from original input HDF5.
    # LFADS output HDF5 usually does NOT contain bin_edges_s; these
    # metadata belong to the HDF5 input produced by the IBL binning script.
    input_h5 = args.input_h5.expanduser().resolve()
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input spike-count HDF5 not found: {input_h5}")
    with h5py.File(input_h5, "r") as f:
        required_input = ("train_recon_data", "valid_recon_data", "bin_edges_s")
        missing = [k for k in required_input if k not in f]
        if missing:
            raise KeyError(f"Missing input HDF5 keys: {missing}; actual keys={list(f.keys())}")
        train = f["train_recon_data"][:].astype(float)
        observed = f["valid_recon_data"][:].astype(float)
        edges = f["bin_edges_s"][:]
        dt = float(np.median(np.diff(edges)))
        time = (edges[:-1] + edges[1:]) / 2

    # LFADS' Poisson readout output represents expected counts per bin.
    with h5py.File(output_h5, "r") as f:
        print("LFADS output HDF5 keys:", list(f.keys()))
        if "valid_output_params" not in f:
            raise KeyError(
                "'valid_output_params' is missing from LFADS output. "
                "Check the exported output keys and your reconstruction distribution."
            )
        predicted = f["valid_output_params"][:].astype(float)

    if observed.shape != predicted.shape:
        raise ValueError(f"Shape mismatch, observed={observed.shape}, predicted={predicted.shape}")
    if observed.shape[1] != len(time):
        raise ValueError("Time coordinate length does not match validation bins")
    if observed.ndim != 3 or train.ndim != 3 or train.shape[2] != observed.shape[2]:
        raise ValueError("Input arrays must have shape Trial x Time x Neuron and same neurons")
    if np.any(observed < 0) or np.any(predicted <= 0) or not np.isfinite(predicted).all():
        raise ValueError("Expected non-negative spike counts and finite positive Poisson predictions")

    print("Validation spike counts:", observed.shape)
    print("Predicted expected counts:", predicted.shape)
    print("Bin size (s):", dt)
    baseline = train.mean(axis=(0, 1), keepdims=True)
    total_bps, per_neuron_bps, counts = poisson_bps(observed, predicted, baseline)
    print(f"Validation BPS versus train-set constant-rate baseline: {total_bps:.4f}")
    for idx, (bps, n) in enumerate(zip(per_neuron_bps, counts), start=1):
        print(f"Neuron {idx}: BPS={bps:.4f}, validation spike count={n:.0f}")
    print("Note: validation spikes were also provided to the encoder; this is held-out-trial reconstruction, NOT spike-held-out forecasting.")

    # PSTH: both observations and posterior predictive means are in count/bin.
    # Convert to Hz and smooth only for the visual comparison.
    observed_psth_hz = gaussian_filter1d(observed.mean(axis=0) / dt, args.sigma_bins, axis=0)
    predicted_psth_hz = gaussian_filter1d(predicted.mean(axis=0) / dt, args.sigma_bins, axis=0)
    n_neurons = observed.shape[2]
    fig, axes = plt.subplots(n_neurons, 1, figsize=(11, max(6, 1.75 * n_neurons)), sharex=True)
    axes = np.atleast_1d(axes)
    for neuron, ax in enumerate(axes):
        ax.plot(time, observed_psth_hz[:, neuron], label="Observed PSTH", linewidth=1.7)
        ax.plot(time, predicted_psth_hz[:, neuron], label="LFADS expected rate", linewidth=1.7)
        ax.axvline(0, linestyle="--", color="gray", alpha=0.65)
        ax.set_ylabel(f"N{neuron + 1}\nHz")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time relative to first movement (s)")
    fig.suptitle("Validation: observed vs LFADS-inferred firing rates", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    psth_path = run_dir / "validation_observed_vs_lfads.png"
    fig.savefig(psth_path, dpi=180)
    plt.close(fig)
    print("Saved:", psth_path)


if __name__ == "__main__":
    main()
