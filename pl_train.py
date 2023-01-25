import os
import sys
import argparse

import torch
torch.set_default_dtype(torch.float32)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import AgentFormerDataModule
from utils.config import Config
from utils.utils import get_timestring
from trainer import AgentFormerTrainer


def main(args):
    # Set global random seed
    cfg = Config(args.cfg)
    pl.seed_everything(cfg.seed)

    plugin = DDPStrategy(find_unused_parameters=False)
    args.devices = torch.cuda.device_count()
    accelerator = 'gpu'
    sanity_val_steps = 1
    lim_train_batch = None
    lim_val_batch = None

    # Initialize data module
    dm = AgentFormerDataModule(cfg, args)
    model = AgentFormerTrainer(cfg, args)
    model.model.set_device(model.device)

    # Initialize trainer
    default_root_dir = os.path.join(args.logs_root, cfg.id)
    if args.checkpoint_path is not None and args.resume:
        resume_from_checkpoint = args.checkpoint_path
        print("LOADING from custom checkpoint:", resume_from_checkpoint)
    else:
        resume_from_checkpoint = None
        print("STARTING new run from scratch")
    # Initialize trainer
    logger = TensorBoardLogger(args.logs_root, name=cfg.id)
    early_stop_cb = EarlyStopping(patience=20, verbose=True, monitor='val/ADE')
    checkpoint_callback = ModelCheckpoint(monitor='val/ADE', save_top_k=3, mode='min', save_last=True,
                                          every_n_epochs=1, dirpath=default_root_dir, filename='h_{epoch:04d}')

    print("LOGGING TO:", default_root_dir)
    trainer = pl.Trainer(check_val_every_n_epoch=5, num_sanity_val_steps=sanity_val_steps,
                         devices=args.devices, strategy=plugin, accelerator=accelerator,
                         log_every_n_steps=50 if lim_train_batch is None else lim_train_batch,
                         limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                         max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                         logger=logger, callbacks=[early_stop_cb, checkpoint_callback],)

    if 'train' in args.mode:
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
    elif 'test' in args.mode or 'val' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_pre')
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--dont_resume', '-dr', dest='resume', action='store_false', default=True)
    parser.add_argument('--checkpoint_path', '-cp', default=None)
    parser.add_argument('--save_viz', '-v', action='store_true', default=False)
    parser.add_argument('--logs_root', default='results')
    args = parser.parse_args()

    time_str = get_timestring()
    print("time str: {}".format(time_str))
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print("torch version : {}".format(torch.__version__))
    print("cudnn version : {}".format(torch.backends.cudnn.version()))
    main(args)
