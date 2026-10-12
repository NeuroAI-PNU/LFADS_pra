"""Reusable spike-count preprocessing helpers for LFADS-style HDF5 files.

No ONE/IBL dependency. Intended for both IBL and other extracellular datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def make_bin_edges(t_before: float, t_after: float, bin_size: float) -> np.ndarray:
    """Time edges relative to each event, e.g. [-.5, ..., 1.0] seconds."""
    if t_before < 0 or t_after <= 0 or bin_size <= 0:
        raise ValueError("Require t_before >= 0, t_after > 0, bin_size > 0")
    n_float = (t_before + t_after) / bin_size
    n = round(n_float)
    if n < 1 or not np.isclose(n, n_float, rtol=0, atol=1e-8):
        raise ValueError("(t_before + t_after) must be divisible by bin_size")
    return np.linspace(-t_before, t_after, n + 1, dtype=float)


def select_ibl_cluster_ids(clusters: Any, region: str = "MOp", quality: str = "good") -> np.ndarray:
    """Use explicit cluster_id values, NEVER row numbers from np.where(mask)."""
    if "cluster_id" not in clusters:
        raise KeyError("Missing clusters['cluster_id']; cannot safely map spike IDs")
    cids = np.asarray(clusters["cluster_id"])
    if cids.ndim != 1 or len(np.unique(cids)) != len(cids):
        raise ValueError("Cluster IDs must be a unique 1-D array")
    mask = np.ones(len(cids), dtype=bool)
    if region.strip().upper() not in {"ALL", "*"}:
        if "acronym" not in clusters:
            raise KeyError("Missing clusters['acronym']; brain region metadata unavailable")
        acronyms = np.asarray(clusters["acronym"]).astype(str)
        if len(acronyms) != len(cids):
            raise ValueError("Region acronym and cluster_id lengths differ")
        mask &= np.char.startswith(acronyms, region)
    if quality == "good":
        if "label" not in clusters:
            raise KeyError("Missing clusters['label']; cannot select good units")
        labels = np.asarray(clusters["label"])
        if len(labels) != len(cids):
            raise ValueError("Quality label and cluster_id lengths differ")
        mask &= labels == 1
    elif quality != "all":
        raise ValueError("quality must be 'good' or 'all'")
    selected = np.sort(cids[mask].astype(np.int64))
    if len(selected) == 0:
        raise ValueError(f"No units found (region={region}, quality={quality})")
    return selected


def bin_spikes_by_trial(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    event_times: np.ndarray,
    cluster_ids: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Return (trial, time_bin, neuron) nonnegative integer spike COUNTS.

    Every event is processed independently. Overlapping windows are allowed in X;
    the train/valid split step below purges overlap at the split boundary.
    """
    times = np.asarray(spike_times, dtype=float)
    spike_ids = np.asarray(spike_clusters)
    events = np.asarray(event_times, dtype=float)
    ids = np.asarray(cluster_ids)
    edges = np.asarray(bin_edges, dtype=float)
    if times.ndim != 1 or spike_ids.shape != times.shape:
        raise ValueError("spike_times and spike_clusters must be 1-D, equal length")
    if events.ndim != 1 or not np.isfinite(events).all() or len(events) == 0:
        raise ValueError("event_times must be nonempty, finite and 1-D")
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("cluster_ids must be nonempty, unique and 1-D")
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")
    if not np.isfinite(times).all() or not np.isfinite(edges).all():
        raise ValueError("Nonfinite spike times or bin edges")

    # Preserve caller's unit order in X while searching using sorted IDs.
    id_order = np.argsort(ids)
    sorted_ids = ids[id_order]
    in_units = np.isin(spike_ids, sorted_ids)
    times = times[in_units]
    spike_ids = spike_ids[in_units]
    sort_idx = np.argsort(times, kind="stable")
    times = times[sort_idx]
    spike_ids = spike_ids[sort_idx]
    unit_columns = id_order[np.searchsorted(sorted_ids, spike_ids)]

    n_trials = len(events)
    n_bins = len(edges) - 1
    n_units = len(ids)
    if n_trials * n_bins * n_units > 1_000_000_000:
        raise MemoryError("Resulting tensor is too large; consider smaller selections")
    X = np.zeros((n_trials, n_bins, n_units), dtype=np.int32)
    for i, t_event in enumerate(events):
        start = t_event + edges[0]
        stop = t_event + edges[-1]
        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, stop, side="right")
        if right == left:
            continue
        relative_times = times[left:right] - t_event
        bins = np.searchsorted(edges, relative_times, side="right") - 1
        # Like np.histogram, include a spike exactly on the final right edge.
        bins[relative_times == edges[-1]] = n_bins - 1
        columns = unit_columns[left:right]
        valid = (bins >= 0) & (bins < n_bins)
        flat = bins[valid] * n_units + columns[valid]
        X[i] = np.bincount(flat, minlength=n_bins * n_units).reshape(n_bins, n_units)
    return X


def chronological_split(
    event_times: np.ndarray,
    bin_edges: np.ndarray,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Chronological train/valid trial split; remove train windows overlapping validation.

    This only purges the train/validation boundary. It does not eliminate overlap
    among trials within one split.
    """
    events = np.asarray(event_times, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    if events.ndim != 1 or len(events) < 2 or not np.isfinite(events).all():
        raise ValueError("Need >= 2 finite events")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    order = np.argsort(events, kind="stable")
    cut = max(1, min(len(events) - 1, int(len(events) * train_fraction)))
    train, valid = order[:cut], order[cut:]
    first_valid_start = np.min(events[valid] + edges[0])
    keep = events[train] + edges[-1] <= first_valid_start
    excluded = int(np.count_nonzero(~keep))
    train = train[keep]
    if not len(train):
        raise ValueError("No train trials after removing cross-split overlap")
    return train, valid, excluded


def save_lfads_h5(
    path: str | Path,
    X: np.ndarray,
    event_times: np.ndarray,
    cluster_ids: np.ndarray,
    bin_edges: np.ndarray,
    trial_ids: np.ndarray | None = None,
    train_fraction: float = 0.8,
    overwrite: bool = False,
    metadata: dict[str, str | int | float] | None = None,
) -> dict[str, Any]:
    """Write LFADS-torch's four required arrays and traceability metadata."""
    X = np.asarray(X)
    events = np.asarray(event_times, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    ids = np.asarray(cluster_ids)
    if X.ndim != 3 or not np.issubdtype(X.dtype, np.integer):
        raise ValueError("X must be a (trials, time_bins, units) integer spike-count array")
    nt, nb, nu = X.shape
    if nt < 2 or nb < 1 or nu < 1 or events.shape != (nt,):
        raise ValueError("X shape and event_times length do not match")
    if len(edges) != nb + 1 or len(ids) != nu:
        raise ValueError("X dimensions disagree with bin_edges or cluster_ids")
    if not np.isfinite(events).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Invalid event_times or bin_edges")
    if np.any(X < 0):
        raise ValueError("Spike counts cannot be negative")
    if trial_ids is None:
        trial_ids = np.arange(nt, dtype=np.int64)
    trial_ids = np.asarray(trial_ids)
    if trial_ids.shape != (nt,):
        raise ValueError("trial_ids must match X trial dimension")

    train_idx, valid_idx, excluded = chronological_split(events, edges, train_fraction)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as h5:
        for name, idx in (("train", train_idx), ("valid", valid_idx)):
            counts = X[idx]
            h5.create_dataset(f"{name}_encod_data", data=counts, compression="gzip")
            h5.create_dataset(f"{name}_recon_data", data=counts, compression="gzip")
            h5.create_dataset(f"{name}_trial_ids", data=trial_ids[idx])
            h5.create_dataset(f"{name}_event_times_s", data=events[idx])
        h5.create_dataset("cluster_ids", data=ids)
        h5.create_dataset("bin_edges_s", data=edges)
        h5.attrs["bin_size_s"] = float(np.mean(np.diff(edges)))
        h5.attrs["split_method"] = "chronological_80_20_with_overlap_purge" if train_fraction == 0.8 else "chronological_with_overlap_purge"
        h5.attrs["excluded_cross_split_trials"] = excluded
        for k, v in (metadata or {}).items():
            if k in h5.attrs:
                raise ValueError(f"Metadata attribute {k} is reserved")
            h5.attrs[k] = v
    return {
        "path": str(path),
        "original_shape": tuple(X.shape),
        "train_shape": tuple(X[train_idx].shape),
        "valid_shape": tuple(X[valid_idx].shape),
        "excluded_cross_split_trials": excluded,
    }
