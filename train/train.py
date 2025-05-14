# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import mmseg.datasets.plantseg115

import mlflow
import mlflow.pytorch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from the latest checkpoint')
    parser.add_argument('--amp', action='store_true', default=False, help='enable AMP training')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--mlflow-exp-name', type=str, default="PlantSeg_Experiments", help="MLflow experiment name")
    parser.add_argument('--mlflow-uri', type=str, default=None, help="MLflow tracking URI (e.g., http://<ip>:5000)")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Enable AMP
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP is already enabled.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', 'AMP only supported with OptimWrapper'
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume
    cfg.resume = args.resume

    # MLflow setup
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp_name)

    mlflow.start_run()
    try:
        # Log hyperparameters
        mlflow.log_params({
            'config': args.config,
            'work_dir': cfg.work_dir,
            'amp': args.amp,
            'resume': args.resume,
            'optimizer': cfg.optim_wrapper.optimizer.type,
            'lr': cfg.optim_wrapper.optimizer.get('lr', 'default'),
            'epochs': cfg.train_cfg.get('max_iters', 'unknown') if hasattr(cfg, 'train_cfg') else 'unknown'
        })

        # Build runner
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # Train
        runner.train()

        # Log best checkpoint
        best_ckpt = None
        for fname in sorted(os.listdir(cfg.work_dir), reverse=True):
            if fname.endswith('.pth') and 'best_mIoU' in fname:
                best_ckpt = os.path.join(cfg.work_dir, fname)
                break

        if best_ckpt and os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path='checkpoints')

    finally:
        mlflow.end_run()


if __name__ == '__main__':
    main()
