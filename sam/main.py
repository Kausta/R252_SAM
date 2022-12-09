import os
import argparse
import json

import torch
import torch.backends.cuda

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import sam.trainers as trainers
from sam.util import load_config, print_config
import sam.data as data

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/debug.yaml", help="Main Config File")
    args = parser.parse_args()

    config = load_config(args.config)
    print(">>> Config:")
    print_config(config)
    print("-----------")

    pl.seed_everything(config.misc.seed)

    datamodule = data.DataModule(config, getattr(data, config.data.dataset_cls))

    wandb_logger = WandbLogger(project=config.project, group=config.group, log_model=False)
    checkpoint_callback = ModelCheckpoint(os.path.join(config.trainer.out_dir, wandb_logger.experiment.name), monitor=config.trainer.pl.monitor, save_last=True, save_top_k=1)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.pl.accerelator,
        devices=config.trainer.pl.devices,
        benchmark=config.trainer.pl.cudnn_benchmark,
        log_every_n_steps=config.trainer.pl.log_freq,
        callbacks=[
            LearningRateMonitor("step"),
            checkpoint_callback
        ],
        precision=config.trainer.pl.precision,
        num_sanity_val_steps=0,
        # TODO: Remove following two lines and do this in the trainer when we switch to manual optimization
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm"
    )

    Model = getattr(trainers, config.trainer.trainer)
    model = Model(config)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()