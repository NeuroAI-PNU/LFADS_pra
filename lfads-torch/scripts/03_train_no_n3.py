#!/usr/bin/env python3
"""Train LFADS-torch on N3-ablated 4-channel data, from scratch on the GPU node.

Requires 02_prepare_no_n3.py to have created configs/model/ibl_m1_no_n3.yaml
and configs/datamodule/ibl_m1_no_n3.yaml. Does not overwrite the original run.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from lfads_torch.run_model import run_model

PROJECT = Path('/home/changhyun/LFADS_pra/lfads-torch')
CONFIG = 'ibl_m1_no_n3'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=PROJECT)
    p.add_argument('--epochs', type=int, default=10, help='Use 10 to match original 10-epoch training')
    p.add_argument('--seed', type=int, default=0, help='Verify original seed to make a fair comparison')
    a = p.parse_args()
    project = a.project_root.expanduser().resolve()
    if a.epochs < 1:
        p.error('--epochs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not visible. Allocate a GPU node with srun before training.')
    for kind in ('model', 'datamodule'):
        cfg = project / 'configs' / kind / f'{CONFIG}.yaml'
        if not cfg.is_file():
            raise FileNotFoundError(f'Missing {cfg}: run 02_prepare_no_n3.py first')
    tag = datetime.now().strftime('%y%m%d_%H%M%S') + '_no_n3'
    run_dir = project / 'runs/lfads-torch-example/ibl_m1_no_n3' / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'train_manifest.json').write_text(
        json.dumps({'model': CONFIG, 'datamodule': CONFIG, 'max_epochs': a.epochs,
                    'seed': a.seed, 'device': str(torch.cuda.get_device_name()),
                    'note': 'This run retrains LFADS with N3 removed, from scratch'},
                   indent=2), encoding='utf-8'
    )
    print('Training run folder:', run_dir)
    print('GPU:', torch.cuda.get_device_name())
    print('IMPORTANT: Check original run_ibl_m1.py for any extra Hydra overrides that differ from these settings.')
    overrides = {
        'datamodule': CONFIG,
        'model': CONFIG,
        'trainer.max_epochs': a.epochs,
        'seed': a.seed,
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        # Uses LFADS-torch's existing, version-matched run_model() API.
        # Do not add posterior_sampling.num_samples: that Hydra key is invalid.
        run_model(overrides=overrides, config_path='../configs/single.yaml')
    finally:
        os.chdir(old_cwd)
    outputs = sorted(run_dir.glob('lfads_output*.h5'))
    print('Training finished. LFADS outputs:', [str(x) for x in outputs])
    if not outputs:
        print('WARNING: No lfads_output*.h5 found. Check posterior sampling settings/logs.')
    print('Next: python scripts/04_compare_no_n3.py --ablated-run', run_dir)


if __name__ == '__main__':
    main()
