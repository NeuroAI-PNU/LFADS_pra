"""Creates SMALL synthetic raw spikes for testing the non-IBL converter.

This is a software example, NOT neural data and NOT an IBL session.
"""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(123)
spike_times = np.sort(rng.uniform(0, 20, size=1200))
spike_clusters = rng.choice([101, 201, 301], size=len(spike_times))
event_times = np.arange(2, 18, 2, dtype=float)
Path("examples").mkdir(exist_ok=True)
np.savez_compressed(
    "examples/demo_raw_spikes.npz",
    spike_times=spike_times,
    spike_clusters=spike_clusters,
    cluster_ids=np.array([101, 201, 301]),
    event_times=event_times,
)
print("Saved examples/demo_raw_spikes.npz")
