"""Inspect and validate an LFADS-torch input .h5 file.

Example:
  python scripts/04_inspect_h5.py lfads_data/m1_<PID>.h5
"""
import argparse

import h5py
import numpy as np


REQUIRED = (
    "train_encod_data",
    "train_recon_data",
    "valid_encod_data",
    "valid_recon_data",
)


def inspect(file):
    print(f"File: {file.filename}")
    for name, value in file.attrs.items():
        print(f"  attr: {name} = {value}")
    for name, obj in file.items():
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:25s} shape={str(obj.shape):20s} dtype={obj.dtype}")
    for name in REQUIRED:
        if name not in file:
            raise KeyError(f"Missing required HDF5 dataset: {name}")
        dataset = file[name]
        if dataset.ndim != 3 or not np.issubdtype(dataset.dtype, np.integer):
            raise ValueError(f"{name}: expected rank-3 integer spike counts")
        if np.any(dataset[:] < 0):
            raise ValueError(f"{name}: found negative spike count")

    for split in ("train", "valid"):
        a, b = file[f"{split}_encod_data"], file[f"{split}_recon_data"]
        if a.shape != b.shape:
            raise ValueError(f"{split}: encoder and reconstruction shapes differ")
        if "cluster_ids" in file and a.shape[2] != len(file["cluster_ids"]):
            raise ValueError(f"{split}: unit count and cluster_ids length disagree")
        if "bin_edges_s" in file and a.shape[1] + 1 != len(file["bin_edges_s"]):
            raise ValueError(f"{split}: time bins and bin_edges_s disagree")
        for suffix in ("trial_ids", "event_times_s"):
            key = f"{split}_{suffix}"
            if key in file and a.shape[0] != len(file[key]):
                raise ValueError(f"{split}: {suffix} length mismatch")
    if file["train_encod_data"].shape[1:] != file["valid_encod_data"].shape[1:]:
        raise ValueError("Train and validation have different time/unit dimensions")
    print("PASS: LFADS-torch's four required HDF5 datasets are valid shape/type.")
    print("NOTE: This does not guarantee a particular LFADS-torch model config works.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path")
    args = p.parse_args()
    with h5py.File(args.h5_path, "r") as f:
        inspect(f)


if __name__ == "__main__":
    main()
