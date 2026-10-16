#!/usr/bin/env python3
"""Per-neuron validation metrics for an existing IBL MOp LFADS-torch run.

No model loading, checkpoint, GPU, or retraining is needed.

Usage:
    python scripts/validate_neuron_metrics.py
    python scripts/validate_neuron_metrics.py --run-dir /path/to/run --input-h5 /path/to/input.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln

DEFAULT_RUN_DIR = Path(
    "/home/changhyun/LFADS_pra/lfads-torch/runs/"
    "lfads-torch-example/ibl_m1/261012_162141_ibl_m1"
)
DEFAULT_INPUT_H5 = Path(
    "/home/changhyun/LFADS_pra/lfads_data/"
    "m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5"
)
EPS = 1e-10


def poisson_log_likelihood(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson log P(y | mu), with y and mu expressed in spikes/bin."""
    mu = np.maximum(mu, EPS)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def temporal_metrics(obs_hz: np.ndarray, pred_hz: np.ndarray) -> tuple[float, float, float, float]:
    """Pearson r, R2, MAE and RMSE for two time-varying PSTHs (Hz)."""
    err = pred_hz - obs_hz
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    obs_ss = float(np.sum((obs_hz - np.mean(obs_hz)) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / obs_ss if obs_ss > EPS else float("nan")
    if np.std(obs_hz) > EPS and np.std(pred_hz) > EPS:
        corr = float(np.corrcoef(obs_hz, pred_hz)[0, 1])
    else:
        corr = float("nan")
    return corr, r2, mae, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--sigma-bins", type=float, default=3.0,
                        help="Gaussian smoothing of trial-averaged PSTH for temporal metrics; 3 bins = 30 ms")
    args = parser.parse_args()

    if args.sigma_bins < 0:
        parser.error("--sigma-bins must be >= 0")
    run_dir = args.run_dir.expanduser().resolve()
    input_h5 = args.input_h5.expanduser().resolve()
    output_files = sorted(run_dir.glob("lfads_output_*.h5"))
    if len(output_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one lfads_output_*.h5 in {run_dir}, found {len(output_files)}"
        )
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")

    with h5py.File(input_h5, "r") as f:
        needed = ("train_recon_data", "valid_recon_data", "bin_edges_s")
        missing = [k for k in needed if k not in f]
        if missing:
            raise KeyError(f"Missing input HDF5 keys: {missing}; available: {list(f.keys())}")
        train_counts = np.asarray(f["train_recon_data"][:], dtype=np.float64)
        valid_counts = np.asarray(f["valid_recon_data"][:], dtype=np.float64)
        edges = np.asarray(f["bin_edges_s"][:], dtype=np.float64)
        cluster_ids = np.asarray(f["cluster_ids"][:]) if "cluster_ids" in f else None

    with h5py.File(output_files[0], "r") as f:
        if "valid_output_params" not in f:
            raise KeyError(
                "LFADS output file has no 'valid_output_params'. "
                f"Available keys: {list(f.keys())}"
            )
        pred_counts = np.asarray(f["valid_output_params"][:], dtype=np.float64)

    if train_counts.ndim != 3 or valid_counts.ndim != 3:
        raise ValueError("Input must be shaped (trial, time_bin, neuron)")
    if pred_counts.shape != valid_counts.shape:
        raise ValueError(f"Shape mismatch: LFADS {pred_counts.shape}, observed {valid_counts.shape}")
    if train_counts.shape[1:] != valid_counts.shape[1:]:
        raise ValueError("Train/validation time bin and neuron dimensions must match")
    if len(edges) != valid_counts.shape[1] + 1:
        raise ValueError(f"bin_edges_s length should be {valid_counts.shape[1]+1}, got {len(edges)}")
    if not np.all(np.isfinite(valid_counts)) or not np.all(np.isfinite(train_counts)):
        raise ValueError("Spike counts contain NaN/Inf")
    if not np.all(np.isfinite(pred_counts)) or np.any(pred_counts < 0):
        raise ValueError("Model expected spike counts must be non-negative and finite")
    if np.any(train_counts < 0) or np.any(valid_counts < 0):
        raise ValueError("Spike counts must be non-negative")

    dt = float(np.median(np.diff(edges)))
    if dt <= 0:
        raise ValueError("Invalid time-bin edges")
    n_trial, n_time, n_neuron = valid_counts.shape

    # LFADS Poisson predictions and observations are both counts per bin.
    # The constant-rate baseline is learned ONLY from training trials.
    train_mean_counts = train_counts.mean(axis=(0, 1))  # (neuron,)
    baseline_counts = np.maximum(train_mean_counts, EPS)

    # Calculate Poisson log-likelihood (per validation neuron), summed over trials and bins.
    ll_model = poisson_log_likelihood(valid_counts, pred_counts).sum(axis=(0, 1))
    ll_baseline = poisson_log_likelihood(valid_counts, baseline_counts[None, None, :]).sum(axis=(0, 1))
    spike_totals = valid_counts.sum(axis=(0, 1))
    bps = np.full(n_neuron, np.nan, dtype=float)
    has_spikes = spike_totals > 0
    bps[has_spikes] = (ll_model[has_spikes] - ll_baseline[has_spikes]) / (
        spike_totals[has_spikes] * np.log(2)
    )

    # Compare trial-averaged observed and LFADS-predicted time courses.
    obs_psth_hz = valid_counts.mean(axis=0) / dt
    pred_psth_hz = pred_counts.mean(axis=0) / dt
    if args.sigma_bins > 0:
        obs_psth_hz = gaussian_filter1d(obs_psth_hz, sigma=args.sigma_bins, axis=0)
        pred_psth_hz = gaussian_filter1d(pred_psth_hz, sigma=args.sigma_bins, axis=0)

    rows = []
    for i in range(n_neuron):
        r, r2, mae, rmse = temporal_metrics(obs_psth_hz[:, i], pred_psth_hz[:, i])
        rows.append({
            "neuron": f"N{i+1}",
            "cluster_id": str(cluster_ids[i]) if cluster_ids is not None and i < len(cluster_ids) else "",
            "valid_spikes": int(spike_totals[i]),
            "observed_mean_hz": float(valid_counts[:, :, i].mean() / dt),
            "predicted_mean_hz": float(pred_counts[:, :, i].mean() / dt),
            "BPS_vs_train_mean": float(bps[i]),
            "PSTH_Pearson_r": r,
            "PSTH_R2": r2,
            "PSTH_MAE_Hz": mae,
            "PSTH_RMSE_Hz": rmse,
        })

    table = pd.DataFrame(rows)
    output_csv = run_dir / "neuron_validation_metrics.csv"
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nValidation shape: {valid_counts.shape} (trials, bins, neurons)")
    print(f"dt = {dt*1000:.2f} ms; smoothing sigma = {args.sigma_bins} bins")
    print("\n=== Per-neuron validation metrics ===")
    print(table.round(4).to_string(index=False))
    if spike_totals.sum() > 0:
        pooled_bps = (ll_model.sum() - ll_baseline.sum()) / (spike_totals.sum() * np.log(2))
        print(f"\nPooled BPS (spike-weighted, all neurons): {pooled_bps:.4f}")
    print(f"\nSaved CSV: {output_csv}")

    # A compact summary plot; computation itself does not require matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = table["neuron"].tolist()
    xs = np.arange(n_neuron)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    items = [
        ("BPS_vs_train_mean", "Poisson BPS vs train mean"),
        ("PSTH_Pearson_r", "Smoothed PSTH Pearson r"),
        ("PSTH_R2", "Smoothed PSTH R2"),
    ]
    for ax, (col, label) in zip(axes, items):
        vals = table[col].to_numpy(dtype=float)
        bars = ax.bar(xs, vals, color=["tab:orange" if v >= 0 else "tab:blue" for v in vals])
        ax.axhline(0, color="black", linewidth=0.75)
        ax.set_xticks(xs, names)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                            xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    output_png = run_dir / "neuron_validation_metrics.png"
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {output_png}")
    print("\nCAUTION: Validation trial spikes were input to LFADS's encoder. These are reconstruction metrics, not held-out-neuron predictive accuracy.")
    print("PSTH r/R2/MAE are computed on trial-averaged, smoothed curves; BPS uses unsmoothed single-trial spike counts.")


if __name__ == "__main__":
    main()
