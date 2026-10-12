"""Run: python -m unittest discover -s tests -v"""
import tempfile
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lfads_data_utils import (
    bin_spikes_by_trial,
    chronological_split,
    make_bin_edges,
    save_lfads_h5,
    select_ibl_cluster_ids,
)


class TestPipeline(unittest.TestCase):
    def test_true_cluster_id_not_row_index(self):
        clusters = {
            "cluster_id": np.array([900, 12, 5, 103]),
            "acronym": np.array(["MOp", "VISp", "MOp", "MOp"]),
            "label": np.array([1, 1, 0, 1]),
        }
        np.testing.assert_array_equal(
            select_ibl_cluster_ids(clusters, "MOp", "good"),
            np.array([103, 900]),
        )

    def test_binning_matches_histogram(self):
        rng = np.random.default_rng(77)
        ids = np.array([33, 5])  # intentional unsorted neuron order
        times = rng.uniform(0, 5, 1000)
        spike_ids = rng.choice([5, 33, 404], size=1000)
        events = np.array([1., 2., 3.])
        edges = make_bin_edges(0.5, 1.0, 0.01)
        X = bin_spikes_by_trial(times, spike_ids, events, ids, edges)
        for i, t in enumerate(events):
            for j, cid in enumerate(ids):
                expected, _ = np.histogram(times[spike_ids == cid] - t, bins=edges)
                np.testing.assert_array_equal(X[i, :, j], expected)

    def test_valid_mask_and_hdf5(self):
        raw_events = np.array([2., float("nan"), 4., 6., float("nan"), 8.])
        valid = np.isfinite(raw_events)
        events = raw_events[valid]
        trial_ids = np.flatnonzero(valid)
        self.assertEqual((len(events), len(trial_ids)), (4, 4))
        edges = make_bin_edges(0.5, 1.0, 0.01)
        X = np.zeros((4, 150, 2), dtype=np.int32)
        X[:, 25, :] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            result = save_lfads_h5(path, X, events, np.array([3, 90]), edges, trial_ids)
            self.assertEqual(result["train_shape"], (3, 150, 2))
            self.assertEqual(result["valid_shape"], (1, 150, 2))
            with h5py.File(path, "r") as f:
                np.testing.assert_array_equal(f["valid_trial_ids"][:], np.array([5]))
                self.assertEqual(f["train_encod_data"].shape, (3, 150, 2))

    def test_overlap_purge(self):
        edges = make_bin_edges(0.5, 1., 0.01)
        train, valid, excluded = chronological_split(np.array([1., 1.8, 3.]), edges, 0.8)
        np.testing.assert_array_equal(train, [0])
        np.testing.assert_array_equal(valid, [2])
        self.assertEqual(excluded, 1)


if __name__ == "__main__":
    unittest.main()
