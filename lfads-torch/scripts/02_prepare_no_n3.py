#!/usr/bin/env python3
"""Prepare a new 4-neuron H5 and LFADS-torch YAML configs, dropping N3=cluster 498.

Does NOT modify original training data or configs. Train/valid trials and times remain fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path('/home/changhyun/LFADS_pra')
PROJECT = ROOT / 'lfads-torch'
ORIG_INPUT = ROOT / 'lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5'
NEW_INPUT = ROOT / 'lfads_data/m1_no_n3_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5'
DROP_CLUSTER_ID = 498
NEW_CONFIG = 'ibl_m1_no_n3'
SPIKE_KEYS = ('train_encod_data', 'train_recon_data', 'valid_encod_data', 'valid_recon_data')


def new_yaml(src: Path, dst: Path, data: dict) -> None:
    if dst.exists():
        raise FileExistsError(f'Refusing overwrite of {dst}; remove manually if necessary')
    with open(dst, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f'Created config: {dst}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-h5', type=Path, default=ORIG_INPUT)
    p.add_argument('--project-root', type=Path, default=PROJECT)
    p.add_argument('--output-h5', type=Path, default=NEW_INPUT)
    p.add_argument('--original-config', default='ibl_m1')
    p.add_argument('--new-config', default=NEW_CONFIG)
    args = p.parse_args()
    inp = args.input_h5.expanduser().resolve()
    out = args.output_h5.expanduser().resolve()
    project = args.project_root.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(inp)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if inp == out:
        raise ValueError('Input and output HDF5 cannot be the same file')

    m_in = project / 'configs/model' / f'{args.original_config}.yaml'
    d_in = project / 'configs/datamodule' / f'{args.original_config}.yaml'
    m_out = project / 'configs/model' / f'{args.new_config}.yaml'
    d_out = project / 'configs/datamodule' / f'{args.new_config}.yaml'
    for path in (m_in, d_in):
        if not path.is_file():
            raise FileNotFoundError(f'Cannot find original configuration: {path}')
    for path in (m_out, d_out):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite config: {path}')

    with open(m_in, encoding='utf-8') as f:
        model = yaml.safe_load(f)
    with open(d_in, encoding='utf-8') as f:
        datamodule = yaml.safe_load(f)
    if not isinstance(model, dict) or not isinstance(datamodule, dict):
        raise TypeError('Original YAML configs must be mappings')

    if int(model.get('encod_data_dim', -1)) != 5:
        raise ValueError(f'Expected model.encod_data_dim = 5, got {model.get("encod_data_dim")}')
    model['encod_data_dim'] = 4
    # LFADS-torch single-session readout with one module, as in official example.
    readout = model.get('readout')
    modules = readout.get('modules') if isinstance(readout, dict) else None
    if not isinstance(modules, list) or len(modules) != 1 or not isinstance(modules[0], dict):
        raise ValueError('Expected model.readout.modules to have one readout module; inspect YAML manually')
    if int(modules[0].get('out_features', -1)) != 5:
        raise ValueError('Expected model.readout.modules[0].out_features=5')
    modules[0]['out_features'] = 4
    # Some custom single-session models use an explicit readin linear map.
    readin = model.get('readin')
    if isinstance(readin, dict) and 'modules' in readin:
        for part in readin['modules']:
            if isinstance(part, dict) and part.get('in_features') == 5:
                part['in_features'] = 4
    if 'recon_data_dim' in model:
        if int(model['recon_data_dim']) != 5:
            raise ValueError('Unexpected model.recon_data_dim; inspect config')
        model['recon_data_dim'] = 4
    if 'datafile_pattern' not in datamodule:
        raise ValueError('Missing datamodule.datafile_pattern')
    datamodule['datafile_pattern'] = str(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(inp, 'r') as fi, h5py.File(out, 'x') as fo:
        if 'cluster_ids' not in fi:
            raise KeyError('Expected cluster_ids in original HDF5')
        cids = np.asarray(fi['cluster_ids'][:])
        if cids.size != 5 or int(cids[2]) != DROP_CLUSTER_ID:
            raise ValueError(f'N3 cluster id mismatch: expected index 2 = {DROP_CLUSTER_ID}, got {cids.tolist()}')
        if any(k not in fi for k in SPIKE_KEYS):
            raise KeyError(f'Expected spike keys: {SPIKE_KEYS}; found {list(fi.keys())}')
        kept = np.array([0, 1, 3, 4], dtype=int)
        for key in fi.keys():
            if key in SPIKE_KEYS:
                orig = np.asarray(fi[key][:])
                if orig.ndim != 3 or orig.shape[2] != 5:
                    raise ValueError(f'{key} shape is {orig.shape}; expected (*,*,5)')
                fo.create_dataset(key, data=orig[:, :, kept], compression='gzip', compression_opts=4)
                for attr, val in fi[key].attrs.items():
                    fo[key].attrs[attr] = val
            elif key == 'cluster_ids':
                fo.create_dataset('cluster_ids', data=cids[kept])
                for attr, val in fi[key].attrs.items():
                    fo[key].attrs[attr] = val
            else:
                fi.copy(key, fo)
                # Avoid silently copying unknown per-neuron arrays that would now be inconsistent.
                if isinstance(fi[key], h5py.Dataset) and fi[key].ndim > 0 and fi[key].shape[-1] == 5:
                    raise ValueError(f'Unknown channel-aligned dataset {key}: {fi[key].shape}; edit preparation script')
        for attr, val in fi.attrs.items():
            fo.attrs[attr] = val
        fo.attrs['ablation'] = 'remove input and reconstruction channel N3'
        fo.attrs['removed_cluster_id'] = DROP_CLUSTER_ID
        fo.attrs['original_input_h5'] = str(inp)

    # Produce config files only after verifying output HDF5 finished.
    try:
        new_yaml(m_in, m_out, model)
        new_yaml(d_in, d_out, datamodule)
    except Exception:
        # Avoid removing a valid 4-neuron dataset; user can inspect/repair config separately.
        raise

    with h5py.File(out, 'r') as f:
        print('\nAblated data prepared:')
        for key in SPIKE_KEYS:
            print(f'  {key}: {f[key].shape}')
        print(f'  cluster_ids: {f["cluster_ids"][:].tolist()}')
    report = {
        'original_h5': str(inp), 'ablated_h5': str(out),
        'removed_neuron': 'N3', 'removed_cluster_id': DROP_CLUSTER_ID,
        'original_cluster_ids': [int(x) for x in cids],
        'remaining_cluster_ids': [int(x) for x in cids[kept]],
        'model_config': str(m_out), 'datamodule_config': str(d_out),
        'kept_channel_indices_0_based': [0, 1, 3, 4],
    }
    with open(out.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nOriginal files were not changed. Next: GPU training with 03_train_no_n3.py')


if __name__ == '__main__':
    main()
