#!/usr/bin/env python3
"""Compare original 5-neuron LFADS with a retrained N3-ablated 4-neuron LFADS.

CPU-only; read existing outputs. Compare held-out reconstruction on the SAME four
neurons and validation trials, and quantify linear predictability of 5-neuron
factors from 4-neuron factors using a held-out subset of validation TRIALS.
The latter is only a latent-representation similarity diagnostic, not causality.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/changhyun/LFADS_pra')
FULL_H5 = ROOT / 'lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5'
ABL_H5 = ROOT / 'lfads_data/m1_no_n3_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5'
FULL_RUN = ROOT / 'lfads-torch/runs/lfads-torch-example/ibl_m1/261012_162141_ibl_m1'
ABL_RUNS_ROOT = ROOT / 'lfads-torch/runs/lfads-torch-example/ibl_m1_no_n3'
KEEP = [0, 1, 3, 4]
EPS = 1e-10


def get_output(run: Path) -> Path:
    paths = sorted(run.glob('lfads_output*.h5'))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected exactly one LFADS output in {run}; found {paths}')
    return paths[0]


def find_abl_run(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir() and list(p.glob('lfads_output*.h5'))]
    if len(dirs) == 0:
        raise FileNotFoundError(f'No completed N3 ablation runs under {root}; specify --ablated-run')
    # Explicitly refuse ambiguous choice, rather than silently selecting the wrong run.
    if len(dirs) != 1:
        raise RuntimeError(f'Multiple N3 ablation runs found: {dirs}. Pass --ablated-run explicitly.')
    return dirs[0]


def poisson_ll(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, EPS)
    return y * np.log(mu) - mu - gammaln(y+1)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((y - np.mean(y))**2)
    return float(1 - np.sum((y-pred)**2)/denom) if denom > EPS else float('nan')


def get_metrics(obs: np.ndarray, pred: np.ndarray,
                train_counts: np.ndarray, dt: float, sigma: float) -> list[dict]:
    baseline = np.maximum(train_counts.mean(axis=(0,1)), EPS)
    ll = poisson_ll(obs, pred).sum(axis=(0,1))
    ll_base = poisson_ll(obs, baseline[None,None,:]).sum(axis=(0,1))
    spike_count = obs.sum(axis=(0,1))
    psth_obs = obs.mean(axis=0)/dt
    psth_pred = pred.mean(axis=0)/dt
    if sigma > 0:
        psth_obs = gaussian_filter1d(psth_obs, sigma=sigma, axis=0)
        psth_pred = gaussian_filter1d(psth_pred, sigma=sigma, axis=0)
    rows = []
    for i in range(obs.shape[2]):
        y = psth_obs[:, i]
        mu = psth_pred[:, i]
        rows.append({
            'BPS': float((ll[i]-ll_base[i]) / (spike_count[i]*np.log(2)))
                   if spike_count[i] > 0 else float('nan'),
            'PSTH_Pearson_r': corr(y, mu),
            'PSTH_R2': r2(y, mu),
            'PSTH_MAE_Hz': float(np.mean(np.abs(y-mu))),
            'observed_mean_Hz': float(obs[:,:,i].mean()/dt),
            'predicted_mean_Hz': float(pred[:,:,i].mean()/dt),
            'num_spikes': int(spike_count[i]),
        })
    return rows


def latent_similarity(f4: np.ndarray, f5: np.ndarray, seed: int = 0) -> dict:
    """Trial-disjoint ridge map from ablated factors to original factors.

    Compare mapping with original-factor time-locked mean, computed on training
    trials only. Latent directions are nonidentifiable across separately trained LFADS.
    """
    if f4.shape[:2] != f5.shape[:2] or f4.ndim != 3 or f5.ndim != 3:
        raise ValueError(f'Incompatible factor shapes: {f4.shape} vs {f5.shape}')
    n = f4.shape[0]
    if n < 10:
        raise ValueError('Need >= 10 validation trials for factor comparison')
    rng = np.random.default_rng(seed)
    ind = rng.permutation(n)
    n_test = max(2, int(round(n * .2)))
    tr, te = ind[n_test:], ind[:n_test]
    xtr, xte = f4[tr].reshape(-1, f4.shape[-1]), f4[te].reshape(-1, f4.shape[-1])
    ytr, yte = f5[tr].reshape(-1, f5.shape[-1]), f5[te].reshape(-1, f5.shape[-1])
    if not np.isfinite(xtr).all() or not np.isfinite(ytr).all():
        raise ValueError('Non-finite LFADS factor array')
    scaler = StandardScaler().fit(xtr)
    model = Ridge(alpha=1.0).fit(scaler.transform(xtr), ytr)
    predicted = model.predict(scaler.transform(xte))
    # Zero-parameter constant baseline and time-varying PSTH-like factor baseline
    global_baseline = np.broadcast_to(ytr.mean(axis=0), yte.shape)
    time_mean = f5[tr].mean(axis=0)
    time_baseline = np.broadcast_to(time_mean, f5[te].shape).reshape(-1, f5.shape[-1])
    sse_pred = float(np.sum((yte-predicted)**2))
    sse_global = float(np.sum((yte-global_baseline)**2))
    sse_time = float(np.sum((yte-time_baseline)**2))
    result = {
        'num_train_validation_trials_for_mapping': int(tr.size),
        'num_test_validation_trials_for_mapping': int(te.size),
        'ridge_factor_R2_vs_train_global_mean': 1-sse_pred/sse_global if sse_global > EPS else float('nan'),
        'ridge_factor_improvement_vs_time_mean': 1-sse_pred/sse_time if sse_time > EPS else float('nan'),
        'time_mean_factor_R2_vs_train_global_mean': 1-sse_time/sse_global if sse_global > EPS else float('nan'),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-input-h5', type=Path, default=FULL_H5)
    p.add_argument('--ablated-input-h5', type=Path, default=ABL_H5)
    p.add_argument('--original-run', type=Path, default=FULL_RUN)
    p.add_argument('--ablated-run', type=Path, default=None)
    p.add_argument('--sigma-bins', type=float, default=3.0)
    p.add_argument('--out-dir', type=Path, default=None)
    a = p.parse_args()
    if a.sigma_bins < 0:
        p.error('--sigma-bins must be >=0')
    ab_run = a.ablated_run or find_abl_run(ABL_RUNS_ROOT)
    ab_run = ab_run.expanduser().resolve()
    dest = (a.out_dir or (ab_run / 'comparison_with_full_model')).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.original_input_h5, 'r') as f:
        full_train = np.asarray(f['train_recon_data'], dtype=float)
        full_valid = np.asarray(f['valid_recon_data'], dtype=float)
        full_ids = np.asarray(f['cluster_ids'])
        edges = np.asarray(f['bin_edges_s'], dtype=float)
        full_trialids = {k: np.asarray(f[k][:]) for k in ('train_trial_ids','valid_trial_ids') if k in f}
    with h5py.File(a.ablated_input_h5, 'r') as f:
        ab_train = np.asarray(f['train_recon_data'], dtype=float)
        ab_valid = np.asarray(f['valid_recon_data'], dtype=float)
        ab_ids = np.asarray(f['cluster_ids'])
        ab_edges = np.asarray(f['bin_edges_s'], dtype=float)
        for key, ids in full_trialids.items():
            if key not in f or not np.array_equal(ids, np.asarray(f[key][:])):
                raise ValueError(f'{key}: trial ordering changed after ablation')
    if full_train.shape[-1] != 5 or full_valid.shape[-1] != 5:
        raise ValueError('Original model input does not have 5 channels')
    if list(map(int, full_ids)) != [487,493,498,500,503]:
        raise ValueError(f'Unexpected original cluster ids {full_ids.tolist()}')
    if list(map(int, ab_ids)) != [487,493,500,503]:
        raise ValueError(f'Unexpected ablated cluster ids {ab_ids.tolist()}')
    if not np.array_equal(full_train[:,:,KEEP], ab_train):
        raise ValueError('Training spike data differ between original and ablation')
    if not np.array_equal(full_valid[:,:,KEEP], ab_valid):
        raise ValueError('Validation spike data differ between original and ablation')
    if not np.array_equal(edges, ab_edges):
        raise ValueError('Time bin edges differ between the models')
    if edges.size != full_valid.shape[1]+1:
        raise ValueError('Time bin edges do not match validation bins')

    with h5py.File(get_output(a.original_run), 'r') as f:
        full_pred = np.asarray(f['valid_output_params'], dtype=float)
        full_factors = np.asarray(f['valid_factors'], dtype=float)
    with h5py.File(get_output(ab_run), 'r') as f:
        ab_pred = np.asarray(f['valid_output_params'], dtype=float)
        ab_factors = np.asarray(f['valid_factors'], dtype=float)
    if full_pred.shape != full_valid.shape or ab_pred.shape != ab_valid.shape:
        raise ValueError(f'Model output mismatch: original {full_pred.shape}, ablated {ab_pred.shape}')
    if np.any(full_pred < 0) or np.any(ab_pred < 0):
        raise ValueError('Model Poisson expected counts must be nonnegative')
    if not np.isfinite(full_pred).all() or not np.isfinite(ab_pred).all():
        raise ValueError('Model output contains NaN/Inf')

    dt = float(np.median(np.diff(edges)))
    times = (edges[:-1]+edges[1:])/2
    m5 = get_metrics(full_valid[:,:,KEEP], full_pred[:,:,KEEP], full_train[:,:,KEEP], dt, a.sigma_bins)
    m4 = get_metrics(ab_valid, ab_pred, ab_train, dt, a.sigma_bins)
    rows = []
    for k, old in enumerate(KEEP):
        item = {'neuron': f'N{old+1}', 'cluster_id': int(ab_ids[k])}
        for key in m5[k]:
            item[f'full5_{key}'] = m5[k][key]
            item[f'noN3_{key}'] = m4[k][key]
            if key in ('BPS', 'PSTH_R2', 'PSTH_Pearson_r', 'PSTH_MAE_Hz'):
                item[f'delta_{key}_noN3_minus_full5'] = m4[k][key] - m5[k][key]
        rows.append(item)
    df = pd.DataFrame(rows)
    csv = dest/'ablation_neuron_metrics.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    result = latent_similarity(ab_factors, full_factors)
    pd.DataFrame([result]).to_csv(dest/'latent_similarity_diagnostic.csv', index=False, encoding='utf-8-sig')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    ix = np.arange(len(rows))
    for ax, metric, name in zip(axes, ('BPS','PSTH_R2'), ('Poisson BPS vs train mean','Smoothed PSTH R2')):
        ax.bar(ix - 0.2, [x[f'full5_{metric}'] for x in rows], 0.4, label='Original 5 neurons')
        ax.bar(ix + 0.2, [x[f'noN3_{metric}'] for x in rows], 0.4, label='N3 removed, retrained')
        ax.set_xticks(ix, df['neuron'].tolist())
        ax.axhline(0, color='black', linewidth=0.7)
        ax.grid(axis='y', alpha=0.2)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.savefig(dest/'ablation_metrics_comparison.png', dpi=180)
    plt.close(fig)

    obs = ab_valid.mean(axis=0)/dt
    f5 = full_pred[:,:,KEEP].mean(axis=0)/dt
    f4 = ab_pred.mean(axis=0)/dt
    if a.sigma_bins:
        obs = gaussian_filter1d(obs, sigma=a.sigma_bins, axis=0)
        f5 = gaussian_filter1d(f5, sigma=a.sigma_bins, axis=0)
        f4 = gaussian_filter1d(f4, sigma=a.sigma_bins, axis=0)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
    for k, ax in enumerate(axes):
        ax.plot(times, obs[:,k], color='black', label='Observed PSTH')
        ax.plot(times, f5[:,k], label='Original 5-neuron LFADS')
        ax.plot(times, f4[:,k], label='LFADS retrained without N3')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_title(f'N{KEEP[k]+1} (cluster {int(ab_ids[k])})')
        ax.set_ylabel('Hz')
        ax.grid(alpha=0.15)
        if k == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time relative to first movement (s)')
    fig.savefig(dest/'ablation_psth_overlay.png', dpi=180)
    plt.close(fig)

    print('\n=== N3 ablation vs original: SAME validation trials, same four neurons ===')
    cols = ['neuron','full5_BPS','noN3_BPS','delta_BPS_noN3_minus_full5',
            'full5_PSTH_R2','noN3_PSTH_R2','delta_PSTH_R2_noN3_minus_full5']
    print(df[cols].round(4).to_string(index=False))
    print('\n=== Linear factor similarity (trial-disjoint; diagnostic only) ===')
    for key, val in result.items():
        print(f'{key}: {val:.4f}' if isinstance(val,float) else f'{key}: {val}')
    print(f'\nResults saved to: {dest}')
    print('WARNING: N3 was removed and model RETRAINED. Differences may be due to model fit, latent orientation,')
    print('sample stochasticity, regularization and small neuronal population; not proof of causal neuronal influence.')
    print('N3 reconstruction itself is not evaluated in the ablated model because it has no N3 output channel.')


if __name__ == '__main__':
    main()
