#!/usr/bin/env python3
"""IBL MOp LFADS: paired comparison of per-trial latent state speed around movement onset.

Reads an existing LFADS-torch valid_factors output and input HDF5 bin_edges_s.
Does not require GPU or retraining. This is exploratory within-session inference.

Usage (from lfads-torch repository root):
    python scripts/movement_onset_paired_test.py

Default comparisons:
    Pre-onset   [-0.30, -0.10) s
    Peri-onset  [-0.05, +0.15) s
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

DEFAULT_RUN_DIR = Path(
    "/home/changhyun/LFADS_pra/lfads-torch/runs/"
    "lfads-torch-example/ibl_m1/261012_162141_ibl_m1"
)
DEFAULT_INPUT_H5 = Path(
    "/home/changhyun/LFADS_pra/lfads_data/"
    "m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5"
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pre-start", type=float, default=-0.30)
    parser.add_argument("--pre-end", type=float, default=-0.10)
    parser.add_argument("--onset-start", type=float, default=-0.05)
    parser.add_argument("--onset-end", type=float, default=0.15)
    parser.add_argument("--pca-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    return parser.parse_args()


def in_half_open_interval(t: np.ndarray, start: float, end: float) -> np.ndarray:
    if not start < end:
        raise ValueError(f"Window start must be less than end: [{start}, {end})")
    # Round to remove bin-edge floating-point errors before half-open comparisons.
    t = np.round(t, 9)
    return (t >= round(start, 9)) & (t < round(end, 9))


def main() -> None:
    args = get_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_h5 = args.input_h5.expanduser().resolve()
    out_dir = (args.out_dir or (run_dir / "movement_onset_paired_test")).expanduser().resolve()
    if args.pca_dim < 1:
        raise ValueError("--pca-dim must be >= 1")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be >= 100")
    output_files = sorted(run_dir.glob("lfads_output_*.h5"))
    if len(output_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one lfads_output_*.h5 in {run_dir}; found {len(output_files)}."
        )
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")

    with h5py.File(output_files[0], "r") as f:
        if "valid_factors" not in f:
            raise KeyError(f"Missing 'valid_factors' in {output_files[0]}; available: {list(f.keys())}")
        factors = np.asarray(f["valid_factors"][:], dtype=np.float64)
    with h5py.File(input_h5, "r") as f:
        if "bin_edges_s" not in f:
            raise KeyError(f"Missing 'bin_edges_s' in {input_h5}; available: {list(f.keys())}")
        edges = np.asarray(f["bin_edges_s"][:], dtype=np.float64)
        trial_ids = np.asarray(f["valid_trial_ids"][:]) if "valid_trial_ids" in f else None

    if factors.ndim != 3:
        raise ValueError(f"valid_factors should have shape (trials, bins, latent factors); got {factors.shape}")
    n_trials, n_bins, n_factors = factors.shape
    if n_trials < 3 or n_bins < 3 or args.pca_dim > n_factors:
        raise ValueError(f"Unexpected factor shape {factors.shape} or --pca-dim {args.pca_dim}")
    if len(edges) != n_bins + 1:
        raise ValueError(f"Time edges length {len(edges)} does not match n_bins+1={n_bins + 1}")
    if not np.isfinite(factors).all() or not np.isfinite(edges).all():
        raise ValueError("NaN or Inf found in valid_factors or bin_edges_s")
    if trial_ids is not None and len(trial_ids) != n_trials:
        raise ValueError("Trial IDs count differs from valid_factors trial count")

    centers = (edges[:-1] + edges[1:]) / 2.0
    dt = np.diff(centers)
    if not np.all(dt > 0):
        raise ValueError("Time-bin centers must be strictly increasing")
    if not np.allclose(dt, 0.01, atol=1e-5, rtol=0):
        print(f"NOTE: inferred median dt = {np.median(dt):.6f} s (not exactly 10 ms)")

    # Match the original visualization: PCA on all validation trials' LFADS factors.
    pca = PCA(n_components=args.pca_dim)
    coords = pca.fit_transform(factors.reshape(-1, n_factors)).reshape(n_trials, n_bins, args.pca_dim)

    # Individual trial displacement magnitudes / time; shape (trial, n_bins-1).
    velocities = np.diff(coords, axis=1) / dt[None, :, None]
    trial_speed = np.linalg.norm(velocities, axis=2)
    speed_time = (centers[:-1] + centers[1:]) / 2.0
    pre_mask = in_half_open_interval(speed_time, args.pre_start, args.pre_end)
    onset_mask = in_half_open_interval(speed_time, args.onset_start, args.onset_end)
    if not pre_mask.any() or not onset_mask.any():
        raise ValueError("Time selection yielded no bins; inspect bin_edges_s")
    if np.any(pre_mask & onset_mask):
        raise ValueError("Pre and peri-onset windows must not overlap")

    # Paired observations: each of 131 trials contributes one value per window.
    pre = trial_speed[:, pre_mask].mean(axis=1)
    peri = trial_speed[:, onset_mask].mean(axis=1)
    diff = peri - pre
    t_stat, p_two = stats.ttest_rel(peri, pre, alternative="two-sided")
    _, p_greater = stats.ttest_rel(peri, pre, alternative="greater")
    sd_diff = float(np.std(diff, ddof=1))
    mean_diff = float(np.mean(diff))
    sem = sd_diff / np.sqrt(n_trials)
    radius = float(stats.t.ppf(0.975, df=n_trials-1) * sem)
    ci_low, ci_high = mean_diff-radius, mean_diff+radius
    dz = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    try:
        wilcoxon_p = float(stats.wilcoxon(diff, alternative="greater").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")

    rng = np.random.default_rng(args.seed)
    bootstrap_means = np.empty(args.n_bootstrap)
    for j in range(args.n_bootstrap):
        idx = rng.integers(0, n_trials, size=n_trials)
        bootstrap_means[j] = diff[idx].mean()
    bootstrap_ci = np.quantile(bootstrap_means, [0.025, 0.975])

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({
        "validation_trial_index": np.arange(n_trials),
        "trial_id": trial_ids if trial_ids is not None else np.arange(n_trials),
        "pre_speed_pc_units_per_s": pre,
        "onset_speed_pc_units_per_s": peri,
        "onset_minus_pre_pc_units_per_s": diff,
    })
    rows.to_csv(out_dir / "trial_level_speed.csv", index=False, encoding="utf-8-sig")

    # Secondary diagnostic: mean-trajectory speed is NOT used in the paired test.
    mean_speed = np.linalg.norm(np.diff(coords.mean(axis=0), axis=0) / dt[:, None], axis=1)
    mean_trial_speed = trial_speed.mean(axis=0)
    speed_table = pd.DataFrame({
        "time_s": speed_time,
        "mean_trial_state_speed": mean_trial_speed,
        "mean_trajectory_speed": mean_speed,
    })
    speed_table.to_csv(out_dir / "speed_timecourse.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "n_trials": n_trials,
        "time_bin_ms": float(np.median(dt)*1000),
        "pca_dimensions": args.pca_dim,
        "pca_variance_explained_sum": float(np.sum(pca.explained_variance_ratio_)),
        "pre_start_s": args.pre_start,
        "pre_end_s": args.pre_end,
        "onset_start_s": args.onset_start,
        "onset_end_s": args.onset_end,
        "pre_speed_bins": int(pre_mask.sum()),
        "onset_speed_bins": int(onset_mask.sum()),
        "pre_mean_speed": float(pre.mean()),
        "onset_mean_speed": float(peri.mean()),
        "mean_paired_difference": mean_diff,
        "paired_t_statistic": float(t_stat),
        "degrees_of_freedom": n_trials - 1,
        "p_two_sided": float(p_two),
        "p_one_sided_onset_greater": float(p_greater),
        "95ci_low_paired_diff": ci_low,
        "95ci_high_paired_diff": ci_high,
        "bootstrap_95ci_low": float(bootstrap_ci[0]),
        "bootstrap_95ci_high": float(bootstrap_ci[1]),
        "cohen_dz": dz,
        "n_trials_onset_greater": int((diff > 0).sum()),
        "wilcoxon_one_sided_p": wilcoxon_p,
    }])
    summary.to_csv(out_dir / "paired_test_summary.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axs[0]
    ax.plot(speed_time, mean_trial_speed, label="Mean trial state speed (paired-test data)", color="#4BA9D8", lw=2)
    ax.plot(speed_time, mean_speed, label="Mean trajectory speed (descriptive)", color="black", lw=1.7)
    ax.axvspan(args.pre_start, args.pre_end, color="#507DB4", alpha=0.17, label="Pre window")
    ax.axvspan(args.onset_start, args.onset_end, color="#E6A051", alpha=0.20, label="Peri-onset window")
    ax.axvline(0, color="#D56535", linestyle="--", lw=1.2)
    ax.set(xlabel="Time from first wheel movement (s)", ylabel="PC units / s", title="LFADS state-space speed")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axs[1]
    rng2 = np.random.default_rng(args.seed)
    for a, b in zip(pre, peri):
        x_jitter = rng2.normal(0, .025)
        ax.plot([x_jitter, 1+x_jitter], [a, b], color="0.5", alpha=0.13, lw=.7)
    ax.scatter(np.zeros(n_trials) + rng2.normal(0, .025, n_trials), pre, color="#507DB4", s=7, alpha=.22)
    ax.scatter(np.ones(n_trials) + rng2.normal(0, .025, n_trials), peri, color="#E6A051", s=7, alpha=.22)
    ax.plot([0,1], [pre.mean(), peri.mean()], marker="o", color="black", lw=2.7, ms=7)
    ax.set_xticks([0,1], ["Pre-onset", "Peri-onset"])
    ax.set(ylabel="Mean speed per trial (PC units / s)", title=f"Paired t-test: t={t_stat:.2f}, p(two-sided)={p_two:.3g}")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out_dir / "movement_onset_paired_test.png", dpi=180)
    plt.close(fig)

    print(f"\nValidation factors: {factors.shape}; PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Pre-onset: [{args.pre_start:+.2f}, {args.pre_end:+.2f}) s, n_speed_bins={pre_mask.sum()}")
    print(f"Peri-onset: [{args.onset_start:+.2f}, {args.onset_end:+.2f}) s, n_speed_bins={onset_mask.sum()}")
    print(f"\nMean pre speed   : {pre.mean():.4f} PC units/s")
    print(f"Mean onset speed : {peri.mean():.4f} PC units/s")
    print(f"Mean paired diff : {mean_diff:+.4f} PC units/s")
    print(f"Paired t-test    : t({n_trials-1})={t_stat:.4f}, p(two-sided)={p_two:.6g}")
    print(f"Onset > Pre test : p(one-sided)={p_greater:.6g}")
    print(f"95% CI diff      : [{ci_low:.4f}, {ci_high:.4f}] PC units/s")
    print(f"Paired Cohen dz  : {dz:.4f}")
    print(f"Wilcoxon (greater) p : {wilcoxon_p:.6g}")
    print(f"Trials onset>pre : {(diff>0).sum()} / {n_trials}")
    print(f"\nSaved outputs in {out_dir}")
    print("  paired_test_summary.csv / trial_level_speed.csv / speed_timecourse.csv")
    print("  movement_onset_paired_test.png")
    print("\nCAUTION: Windows were chosen after inspecting the onset-aligned curve; treat p-values as exploratory.")
    print("CAUTION: Trial-level paired t-test does not correct for serial dependence between successive trials.")
    print("CAUTION: Inference is within a single mouse/session; not population-level generalization.")


if __name__ == "__main__":
    main()
