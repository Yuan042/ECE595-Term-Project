from pytorch_lightning.callbacks import ModelCheckpoint, Timer
import os.path as osp
from argparse import ArgumentParser

import os
import random
import numpy as np

from mmcv import Config
from models import MODELS
from dataloaders import build_dataset
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str,
                        default='test_', help='experiment name')
    parser.add_argument('--num_gpus', type=int,
                        default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    return parser.parse_args()


if __name__ == '__main__':

    # 1) 解析參數 & 讀 config
    args = parse_args()
    cfg = Config.fromfile(osp.join(args.config))
    print(f'Now training with {args.config}...')

    # 2) seed
    seed_everything(args.seed)

    # 3) dataloader
    dataset = build_dataset(
        cfg.dataset, cfg.model.eval_mode, split='train_val')

    cpu = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
    num_workers = max(1, min(cpu - 1, int(getattr(cfg, "workers_per_gpu", 4))))

    def _worker_init_fn(worker_id):
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed)
        random.seed(base_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        dataset['train'],
        batch_size=cfg.imgs_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    val_loaders = []
    checkpoint_callbacks = []   # 只在這裡初始化一次

    # 4) 決定 work_dir（優先用 cfg.runtime.work_dir，否則用 CLI out_dir/exp_name）
    if hasattr(cfg, "runtime") and hasattr(cfg.runtime, "work_dir") and cfg.runtime.work_dir:
        work_dir = cfg.runtime.work_dir
    else:
        work_dir = osp.join(args.out_dir, args.exp_name)
    os.makedirs(work_dir, exist_ok=True)
    print(f"[train] work_dir = {work_dir}")

    # 5) checkpoints 目錄
    ckpt_dir = osp.join(work_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 6) 建 callbacks
    save_every = getattr(cfg, "save_every_n_steps", 500)
    periodic_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}-{step:08d}",
        save_last=True,
        every_n_train_steps=save_every,
        save_on_train_epoch_end=True,
        save_weights_only=False,
        save_top_k=-1,
    )

    # from datetime import timedelta
    # timer = Timer(duration=timedelta(hours=3, minutes=55))

    checkpoint_callbacks = [periodic_ckpt]

    # 7) 如果有 val，就建 val_loader + 最佳 ckpt（監控 val_loss）
    if 'depth' in cfg.model.eval_mode:
        val_loader_ = DataLoader(
            dataset['val']['depth'],
            batch_size=cfg.imgs_per_gpu,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            generator=g,
        )
        val_loaders.append(val_loader_)
        best_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='best-{epoch:02d}-{step}',
        )
        checkpoint_callbacks.append(best_ckpt)

    print('{} samples found for training'.format(len(train_loader)))
    for idx, val_loader in enumerate(val_loaders):
        print('{} samples found for validatioin set {}'.format(len(val_loader), idx))

    # 8) 構建 model
    model = MODELS.build(name=cfg.model.name, option=cfg)

    # 9) 載入「純權重」再重新訓練（與續訓不同）
    #    把 --ckpt_path 拆成兩個參數比較不會混淆：
    #    --resume_from 只給「完整 ckpt」；--weights_path 只給「weights-only」
    import argparse as _ap
    # 临时兼容：若不改 argparse，可把下面兩行改回 args.weights_path / args.resume_from
    weights_path = None
    resume_from = None
    # 若之後在 argparse 加：
    # parser.add_argument('--weights_path', type=str, default=None)
    # parser.add_argument('--resume_from', type=str, default=None)
    # 再把這兩行改成：
    # weights_path = args.weights_path
    # resume_from = args.resume_from
    if weights_path:
        print(f'load weights from {weights_path}')
        sd = torch.load(weights_path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        model.load_state_dict(sd, strict=False)

    # 10) Trainer
    strategy = DDPStrategy(find_unused_parameters=False) if (
        args.num_gpus and args.num_gpus > 1) else "auto"
    trainer = Trainer(
        strategy=strategy,
        accelerator="gpu" if (args.num_gpus and args.num_gpus > 0) else "auto",
        accumulate_grad_batches=4,
        devices=args.num_gpus or 1,
        default_root_dir=work_dir,
        num_nodes=1,
        num_sanity_val_steps=0,
        max_epochs=getattr(cfg, "total_epochs", 100),
        check_val_every_n_epoch=1,
        limit_train_batches=getattr(cfg, "batch_lim_per_epoch", 1.0),
        callbacks=checkpoint_callbacks,     # 這次包含 step/epoch/last/timer/best
        benchmark=True,                     # 保留加速設定
        precision=32,
    )

    # 11) 續訓（完整 ckpt）；優先用命令列指定，其次自動撿 last.ckpt
    last_ckpt = osp.join(ckpt_dir, "last.ckpt")
    auto_resume = last_ckpt if os.path.isfile(last_ckpt) else None

    #  正確邏輯：先用 CLI --ckpt_path，否則用 auto_resume
    import inspect

    # 先用命令列 --ckpt_path，其次自動撿 last.ckpt
    ckpt_to_resume = args.ckpt_path if getattr(
        args, "ckpt_path", None) else auto_resume
    if ckpt_to_resume and not os.path.isfile(ckpt_to_resume):
        print(
            f"[RESUME] ckpt not found -> {ckpt_to_resume}; start from scratch")
        ckpt_to_resume = None
    else:
        print(f"[RESUME] will resume from: {ckpt_to_resume}")

    # 兼容不同 PL 版本的參數名
    sig = inspect.signature(trainer.fit)
    if "ckpt_path" in sig.parameters:
        trainer.fit(model, train_loader, val_dataloaders=val_loaders,
                    ckpt_path=ckpt_to_resume)
    else:
        # 舊版 (<1.7) 走 resume_from_checkpoint
        if ckpt_to_resume:
            print("[RESUME] using Trainer(..., resume_from_checkpoint=...) for old PL")
            trainer = type(trainer)(
                **{**trainer.__dict__['_trainer_kwargs'], "resume_from_checkpoint": ckpt_to_resume})
        trainer.fit(model, train_loader, val_dataloaders=val_loaders)
