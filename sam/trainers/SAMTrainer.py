import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pytorch_lightning as pl
import lpips

import sam.nn as snn
import sam.nn.functional as SF
import sam.models as models

from sam.util import ConfigType
from sam.nn.optim import get_cosine_schedule_with_warmup

from torchmetrics import Accuracy

__all__ = ["SAMTrainer"]


class SAMTrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()

        self.save_hyperparameters(config)

        ModelCls = getattr(models, config.model.model_cls)
        self.model = ModelCls(self.hparams)
        print(self.model)
        
        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=0.0 if config.loss.label_smoothing is None else config.loss.label_smoothing)
        self.acc = nn.ModuleDict({f"{phase}_acc": Accuracy(task='multiclass', num_classes=config.model.num_outputs) for phase in ["train", "val", "test"]})

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred

    def loss(self, x: torch.Tensor, y: torch.Tensor, phase="train"):
        pred = self.forward(x)

        loss_ce = self.loss_ce(pred, y)
        loss = loss_ce

        self.acc[f"{phase}_acc"](pred, y)

        return {
            "loss": loss,
            "loss_ce": loss_ce
        }

    def log_all(self, out, phase="train"):
        on_step = (phase == "train")
        for k, v in out.items():
            self.log(f'{phase}/{k}', v, on_step=on_step, on_epoch=True)
        self.log(f"{phase}/accuracy", self.acc[f"{phase}_acc"], on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt = self.optimizers()
        lr_sched = self.lr_schedulers()

        opt.zero_grad()
        out = self.loss(x, y, "train")
        self.manual_backward(out["loss"])
        if self.hparams.sam.use_sam:
            def closure():
                SF.disable_running_stats(self.model)
                second_step_loss = self.loss_ce(self.forward(x), y)
                self.manual_backward(second_step_loss)
                self.clip_gradients(
                    optimizer=opt, 
                    gradient_clip_val=self.hparams.optimizer.clip_grad_norm,
                    gradient_clip_algorithm="norm")
                return second_step_loss
            opt.step(closure=closure)
            SF.enable_running_stats(self.model)
        else:
            self.clip_gradients(
                optimizer=opt, 
                gradient_clip_val=self.hparams.optimizer.clip_grad_norm,
                gradient_clip_algorithm="norm")
            opt.step()

        if lr_sched is not None and self.hparams.optimizer.sched_interval == "step":
            lr_sched.step()
        
        self.log_all(out, "train")

    def training_epoch_end(self, outputs) -> None:
        lr_sched = self.lr_schedulers()

        if lr_sched is not None and self.hparams.optimizer.sched_interval == "epoch":
            lr_sched.step()

        # save checkpoint with pytorch-lightning through WandbLogger callback
        if self.trainer.current_epoch % 1 == 0:
            self.trainer.save_checkpoint(f"checkpoint-{self.trainer.current_epoch}.ckpt")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.loss(x, y, "val")
        self.log_all(out, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.loss(x, y, "test")
        self.log_all(out, "test")

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer
        opt_name = opt_params.optimizer

        if self.hparams.sam.use_sam:
            if opt_name == "sgd":
                base_opt = torch.optim.SGD
                kwargs = {
                    "lr": opt_params.lr,
                    "momentum": opt_params.momentum,
                    "weight_decay": opt_params.wd,
                    "nesterov": opt_params.nesterov
                }
            else:
                raise ValueError(f"Unknown optimizer {opt_name}")
            opt = snn.SAM(self.model.parameters(), base_opt, rho=self.hparams.sam.rho, adaptive=self.hparams.sam.adaptive, **kwargs)
        elif opt_name == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=opt_params.lr,
                             momentum=opt_params.momentum,
                             weight_decay=opt_params.wd,
                             nesterov=opt_params.nesterov)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        if opt_params.scheduler is None:
            return opt

        estimated_stepping_batches = self.trainer.estimated_stepping_batches
        if opt_params.scheduler == "cosine":
            sched = get_cosine_schedule_with_warmup(
                opt, opt_params.warmup_epochs * self.trainer.num_training_batches, estimated_stepping_batches)
        else:
            raise ValueError(f"Unknown scheduler {opt_params.scheduler}")

        return [opt], [{
            "scheduler": sched,
            "interval": opt_params.sched_interval
        }]

