import sys
import os
import argparse
import mlflow
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook

# Ensure dataset registration is triggered
sys.path.append('./')
import mmseg.datasets.plantseg_dataset


class InlineMlflowHook(Hook):
    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % self.interval == 0:
            metrics = runner.message_hub.get_scalar("train")
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=runner.iter)

    def after_val_epoch(self, runner, metrics=None):
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(f"val/{k}", v, step=runner.epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train with MLflow')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='dir to save logs and models')
    parser.add_argument('--mlflow-uri', default='http://129.114.27.198:8000')
    parser.add_argument('--mlflow-bucket', default='mmseg-models')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision (AMP)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.amp:
        cfg.setdefault('train_cfg', {})
        cfg.amp = True

    if args.work_dir:
        cfg.work_dir = args.work_dir

    # Start MLflow logging
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("SegNeXt_PlantSeg")

    with mlflow.start_run():
        mlflow.log_param("config", os.path.basename(args.config))
        if 'embed_dims' in cfg.model['backbone']:
            mlflow.log_param("model", cfg.model['backbone']['embed_dims'])

        # Build and register MLflow logging hook
        runner = Runner.from_cfg(cfg)
        runner.register_hook(InlineMlflowHook(interval=50))

        # Start training
        runner.train()

        # Log final checkpoint
        final_ckpt = None
        for fname in sorted(os.listdir(cfg.work_dir), reverse=True):
            if fname.endswith('.pth') and 'iter_' in fname:
                final_ckpt = os.path.join(cfg.work_dir, fname)
                break

        if final_ckpt and os.path.exists(final_ckpt):
            mlflow.log_artifact(final_ckpt, artifact_path='checkpoints')

        # Optional: log all artifacts in work_dir
        # mlflow.log_artifacts(cfg.work_dir)


if __name__ == '__main__':
    main()

