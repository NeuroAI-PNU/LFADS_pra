#!/usr/bin/env python3
"""Analyze N3 vs PC1 and all five neurons vs LFADS factors on validation trials.

CPU only. Does not re-train the model or change the existing output files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

ROOT = Path('/home/changhyun/LFADS_pra')
DEFAULT_INPUT = ROOT / 'lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5'
DEFAULT_RUN = ROOT / 'lfads-torch/runs/lfads-torch-example/ibl_m1/261012_162141_ibl_m1'


def output_h5(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob('lfads_output*.h5'))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one lfads_output*.h5 under {run_dir}, found: {paths}')
    return paths[0]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 3:
        return float('nan')
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-h5', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--full-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--sigma-bins', type=float, default=3.0,
                   help='Gaussian smoothing on individual trials along time axis; 3 bins = 30 ms')
    p.add_argument('--out-dir', type=Path, default=None)
    a = p.parse_args()
    if a.sigma_bins < 0:
        p.error('--sigma-bins must be >=0')
    run_dir = a.full_run.resolve()
    dest = (a.out_dir or (run_dir / 'n3_pc1_analysis')).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.input_h5, 'r') as f:
        counts = np.asarray(f['valid_recon_data'], dtype=float)
        edges = np.asarray(f['bin_edges_s'], dtype=float)
        ids = np.asarray(f['cluster_ids'])
    with h5py.File(output_h5(run_dir), 'r') as f:
        factors = np.asarray(f['valid_factors'], dtype=float)
        pred_counts = np.asarray(f['valid_output_params'], dtype=float)

    if counts.ndim != 3 or counts.shape[2] != 5:
        raise ValueError(f'Expected 5 neurons (trials,time,5), got {counts.shape}')
    if factors.ndim != 3 or factors.shape[:2] != counts.shape[:2]:
        raise ValueError(f'Factors {factors.shape} do not match counts {counts.shape}')
    if pred_counts.shape != counts.shape:
        raise ValueError(f'Predictions {pred_counts.shape} do not match counts {counts.shape}')
    if edges.size != counts.shape[1]+1 or not np.all(np.diff(edges) > 0):
        raise ValueError('bin_edges_s invalid')
    if int(ids[2]) != 498:
        raise ValueError(f'N3 should be cluster 498, got {ids.tolist()}')
    if not all(np.isfinite(x).all() for x in (counts, factors, pred_counts)):
        raise ValueError('Non-finite values in input or outputs')

    dt = float(np.median(np.diff(edges)))
    time = (edges[1:] + edges[:-1]) / 2
    trials, bins, neurons = counts.shape
    # PCA on validation LFADS factors, matching the existing plot_m1_factors workflow.
    pca = PCA(n_components=min(3, factors.shape[2]))
    coords = pca.fit_transform(factors.reshape(-1, factors.shape[2])).reshape(trials, bins, -1)
    pc1 = coords[:, :, 0]
    # Trial-wise Gaussian smoothing is for comparison only. LFADS factors not smoothed here.
    observed_hz = counts / dt
    inferred_hz = pred_counts / dt
    if a.sigma_bins:
        observed_hz = gaussian_filter1d(observed_hz, sigma=a.sigma_bins, axis=1)
        inferred_hz = gaussian_filter1d(inferred_hz, sigma=a.sigma_bins, axis=1)
    mean_pc1 = pc1.mean(axis=0)
    pc1_residual = pc1 - mean_pc1[None, :]

    rows = []
    for i in range(neurons):
        neuron_obs = observed_hz[:, :, i]
        neuron_inferred = inferred_hz[:, :, i]
        obs_residual = neuron_obs - neuron_obs.mean(axis=0)[None, :]
        inf_residual = neuron_inferred - neuron_inferred.mean(axis=0)[None, :]
        per_trial_r = np.array([corr(neuron_obs[j], pc1[j]) for j in range(trials)])
        rows.append({
            'neuron': f'N{i+1}', 'cluster_id': int(ids[i]),
            'observed_PSTH_vs_PC1_r': corr(neuron_obs.mean(axis=0), mean_pc1),
            'inferred_PSTH_vs_PC1_r': corr(neuron_inferred.mean(axis=0), mean_pc1),
            'observed_trial_residual_vs_PC1_r': corr(obs_residual, pc1_residual),
            'inferred_trial_residual_vs_PC1_r': corr(inf_residual, pc1_residual),
            'observed_per_trial_r_median': float(np.nanmedian(per_trial_r))
                if np.isfinite(per_trial_r).any() else float('nan'),
        })
    df = pd.DataFrame(rows)
    out_csv = dest / 'neuron_pc1_correlations.csv'
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # Factor relationships. NOTE: factor orientation/sign is not identifiable.
    fac_rows = []
    for i in range(neurons):
        for k in range(factors.shape[2]):
            obs = observed_hz[:, :, i]
            factor = factors[:, :, k]
            fac_rows.append({
                'neuron': f'N{i+1}', 'cluster_id': int(ids[i]), 'factor': f'F{k+1}',
                'PSTH_Pearson_r': corr(obs.mean(axis=0), factor.mean(axis=0)),
                'trial_residual_Pearson_r': corr(obs - obs.mean(axis=0)[None, :],
                                                 factor - factor.mean(axis=0)[None, :]),
            })
    fac_df = pd.DataFrame(fac_rows)
    fac_df.to_csv(dest / 'neuron_factor_correlations.csv', index=False, encoding='utf-8-sig')

    # Plot N3 and PC1 on normalized scales; magnitude and units are not comparable.
    n3 = observed_hz[:, :, 2].mean(axis=0)
    n3_inf = inferred_hz[:, :, 2].mean(axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    ax = axes[0, 0]
    ax.plot(time, zscore(n3), label='N3 observed PSTH (z)')
    ax.plot(time, zscore(n3_inf), label='N3 LFADS inferred (z)', alpha=0.8)
    ax.plot(time, zscore(mean_pc1), label='Mean PC1 (z)', linewidth=2)
    ax.axvline(0, linestyle='--', color='gray')
    ax.set(xlabel='Time relative to first movement (s)', ylabel='Standardized amplitude',
           title='N3 firing pattern vs mean PC1 trajectory')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    labels = df['neuron'].tolist()
    xs = np.arange(neurons)
    w = 0.4
    ax.bar(xs-w/2, df['observed_PSTH_vs_PC1_r'], w, label='Observed PSTH vs PC1')
    ax.bar(xs+w/2, df['observed_trial_residual_vs_PC1_r'], w,
           label='Trial residual vs PC1')
    ax.set_xticks(xs, labels)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set(ylim=(-1.05, 1.05), ylabel='Pearson r', title='Time average vs trial-to-trial association')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.scatter(mean_pc1, n3, c=time, cmap='coolwarm', s=16)
    ax.set(xlabel='Trial-mean PC1 (arbitrary units)', ylabel='N3 observed PSTH (Hz)',
           title=f'N3 PSTH vs PC1: r={corr(n3, mean_pc1):.3f}')
    ax = axes[1, 1]
    ax.bar(np.arange(len(pca.explained_variance_ratio_))+1,
           pca.explained_variance_ratio_*100)
    ax.set(xlabel='Principal component', ylabel='Explained factor variance (%)',
           title='PCA on 4D LFADS factors (validation trials)')
    fig.savefig(dest / 'n3_pc1_summary.png', dpi=180)
    plt.close(fig)

    heat = fac_df.pivot(index='neuron', columns='factor', values='PSTH_Pearson_r')
    heat2 = fac_df.pivot(index='neuron', columns='factor', values='trial_residual_Pearson_r')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, data, title in zip(axes, (heat, heat2),
                               ('PSTH vs LFADS factors', 'Trial residual vs LFADS factors')):
        im = ax.imshow(data.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(factors.shape[2]), data.columns.tolist())
        ax.set_yticks(range(neurons), data.index.tolist())
        ax.set_title(title)
        for ii in range(neurons):
            for kk in range(factors.shape[2]):
                ax.text(kk, ii, f'{data.iloc[ii,kk]:.2f}', ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(dest / 'neuron_factor_heatmaps.png', dpi=180)
    plt.close(fig)

    print('\n=== N3 / PC1 analysis ===')
    print(f'Input {counts.shape}, factors {factors.shape}, PCA explained variance {pca.explained_variance_ratio_}')
    print(df.round(4).to_string(index=False))
    print(f'\nOutputs saved to: {dest}')
    print('Interpretation: sign of PC1/Factors is arbitrary; association is not causal contribution.')
    print('PSTH correlations may reflect shared movement-aligned temporal trends.')


if __name__ == '__main__':
    main()
