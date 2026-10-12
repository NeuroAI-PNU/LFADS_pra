"""End-to-end IBL probe PID -> LFADS-torch HDF5 spike-count file.

Example:
  python scripts/02_ibl_to_h5.py --pid 0ffa052a-fe3f-46c2-999d-43bd56b7826e
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfads_data_utils import (
    bin_spikes_by_trial,
    make_bin_edges,
    save_lfads_h5,
    select_ibl_cluster_ids,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", required=True, help="Probe insertion UUID from IBL")
    p.add_argument("--region", default="MOp", help="Region prefix or ALL")
    p.add_argument("--quality", choices=["good", "all"], default="good")
    p.add_argument("--event", default="firstMovement_times", help="IBL trials field")
    p.add_argument("--revision", default="2024-05-06", help="Spike-sorting revision or 'latest'")
    p.add_argument("--before", type=float, default=0.5, help="Seconds before event")
    p.add_argument("--after", type=float, default=1.0, help="Seconds after event")
    p.add_argument("--bin-size", type=float, default=0.010, help="Seconds per time bin")
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--out-dir", type=Path, default=Path("lfads_data"))
    p.add_argument("--overwrite", action="store_true", help="Replace existing output (use carefully)")
    args = p.parse_args()

    from one.api import ONE
    from brainbox.io.one import SpikeSortingLoader

    print(f"Loading PID: {args.pid}")
    one = ONE(base_url="https://openalyx.internationalbrainlab.org")
    ssl = SpikeSortingLoader(pid=args.pid, one=one)
    load_kwargs = {} if args.revision.lower() == "latest" else {"revision": args.revision}
    spikes, clusters, channels = ssl.load_spike_sorting(**load_kwargs)
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    cluster_ids = select_ibl_cluster_ids(clusters, args.region, args.quality)
    eid, pname = one.pid2eid(args.pid)
    trials = one.load_object(eid, "trials")
    if args.event not in trials:
        available = [str(k) for k in trials.keys() if "time" in str(k).lower()]
        raise KeyError(f"trials['{args.event}'] missing; available time fields: {available}")
    raw_events = np.asarray(trials[args.event], dtype=float)
    if raw_events.ndim != 1:
        raise ValueError("IBL trial events must be 1-D")
    valid_trials = np.isfinite(raw_events)
    event_times = raw_events[valid_trials]
    trial_ids = np.flatnonzero(valid_trials)

    bin_edges = make_bin_edges(args.before, args.after, args.bin_size)
    X = bin_spikes_by_trial(
        spike_times=np.asarray(spikes["times"]),
        spike_clusters=np.asarray(spikes["clusters"]),
        event_times=event_times,
        cluster_ids=cluster_ids,
        bin_edges=bin_edges,
    )
    # Trial alignment is constructed from ONE mask and cannot silently mismatch.
    assert X.shape[0] == len(event_times) == len(trial_ids)
    assert X.shape[2] == len(cluster_ids)
    output = args.out_dir / f"m1_{args.pid}.h5" if args.region == "MOp" else args.out_dir / f"{args.region}_{args.pid}.h5"
    summary = save_lfads_h5(
        path=output,
        X=X,
        event_times=event_times,
        cluster_ids=cluster_ids,
        bin_edges=bin_edges,
        trial_ids=trial_ids,
        train_fraction=args.train_fraction,
        overwrite=args.overwrite,
        metadata={
            "pid": str(args.pid),
            "eid": str(eid),
            "probe": str(pname),
            "region": args.region,
            "quality": args.quality,
            "event": args.event,
            "revision": args.revision,
            "source": "IBL ONE / SpikeSortingLoader",
        },
    )
    print(f"Total original trials: {len(raw_events)}")
    print(f"Valid event trials: {len(event_times)}")
    print(f"Selected unit IDs: {cluster_ids.tolist()}")
    if len(cluster_ids) < 10:
        print("WARNING: <10 units; useful for a pipeline test, limited for population dynamics.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
