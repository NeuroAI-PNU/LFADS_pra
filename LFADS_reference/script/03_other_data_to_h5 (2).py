"""Convert a non-IBL NumPy .npz dataset to LFADS-torch HDF5.

Supported input formats (all times in SECONDS):

A. Prebinned counts .npz:
   X                 (trials, time_bins, units) integer spike COUNTS
   event_times       (trials,) global recording time for each trial
   cluster_ids       (units,) (optional, defaults to 0..n_units-1)
   trial_ids         (trials,) (optional)
   bin_edges         (time_bins+1,) relative to event (optional; else CLI args)

B. Raw spike times .npz:
   spike_times       (spikes,) global recording time in seconds
   spike_clusters    (spikes,) corresponding neuron IDs
   event_times       (trials,) global recording time for each trial
   cluster_ids       (units,) requested neuron IDs; optional unique(spike_clusters)
   trial_ids         (trials,) (optional)

Example:
  python scripts/03_other_data_to_h5.py --input examples/my_recording.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfads_data_utils import bin_spikes_by_trial, make_bin_edges, save_lfads_h5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--before", type=float, default=0.5)
    p.add_argument("--after", type=float, default=1.0)
    p.add_argument("--bin-size", type=float, default=0.010)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with np.load(args.input, allow_pickle=False) as npz:
        if "event_times" not in npz.files:
            raise KeyError("Missing event_times in input .npz")
        events = np.asarray(npz["event_times"], dtype=float)
        if not np.isfinite(events).all():
            raise ValueError("event_times has NaN/inf; filter events AND X together first")
        trial_ids = npz["trial_ids"] if "trial_ids" in npz.files else np.arange(len(events))
        if "X" in npz.files:
            X = np.asarray(npz["X"])
            ids = npz["cluster_ids"] if "cluster_ids" in npz.files else np.arange(X.shape[2])
            edges = npz["bin_edges"] if "bin_edges" in npz.files else make_bin_edges(
                args.before, args.after, args.bin_size
            )
            data_kind = "prebinned_counts"
        else:
            required = {"spike_times", "spike_clusters"}
            missing = required - set(npz.files)
            if missing:
                raise KeyError(f"Missing .npz keys for raw spikes: {sorted(missing)}")
            spike_times = np.asarray(npz["spike_times"])
            spike_clusters = np.asarray(npz["spike_clusters"])
            ids = npz["cluster_ids"] if "cluster_ids" in npz.files else np.unique(spike_clusters)
            ids = np.sort(np.asarray(ids))
            edges = make_bin_edges(args.before, args.after, args.bin_size)
            X = bin_spikes_by_trial(spike_times, spike_clusters, events, ids, edges)
            data_kind = "raw_spikes"

    output_path = args.output if args.output is not None else Path("lfads_data") / (args.input.stem + ".h5")
    summary = save_lfads_h5(
        output_path, X, events, ids, edges, trial_ids,
        train_fraction=args.train_fraction,
        overwrite=args.overwrite,
        metadata={"source": str(args.input.resolve()), "input_format": data_kind},
    )
    print("WARNING: Check that event_times and X refer to the SAME trials in the SAME order.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
