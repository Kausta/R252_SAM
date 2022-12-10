import torch
import torch.nn as nn
import sam.models as models

from torchcontrib.optim import SWA 
from sam.util import ConfigType
from sam.nn.optim import get_cosine_schedule_with_warmup

import pytorch_lightning as pl

class SWATrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()
        self.save_hyperparameters(config)

        ModelCls = getattr(models, config.model.model_cls)
        self.model = ModelCls(self.hparams)
        print(self.model)

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=0.0 if config.loss.label_smoothing is None else config.loss.label_smoothing)
        self.acc = nn.ModuleDict({f"{phase}_acc": Accuracy(task='multiclass', num_classes=config.model.num_outputs) for phase in ["train", "val", "test"]})
    
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
    
    def configure_optimizers(self):
        opt_params = self.hparams.optimizer
        opt_name = opt_params.optimizer

        base_opt = torch.optim.SGD(self.model.parameters(), 
                                    lr=opt_params.lr, 
                                    momentum=opt_params.momentum,
                                    weight_decay=opt_params.wd,
                                    nesterov=opt_params.nesterov)
        
        opt = SWA(base_opt, swa_start=opt_params.swa_start, swa_freq=opt_params.swa_freq, swa_lr=opt_params.swa_lr)

        if opt_params.scheduler is None:
            return opt
        
        estimated_stepping_batches = self.trainer.estimated_stepping_batches
        if opt_params.scheduler == 'cosine':
            sched = get_cosine_schedule_with_warmup(
                opt, opt_params.warmup_epochs * self.trainer.num_training_batches, estimated_stepping_batches)
        else:
            raise ValueError(f"Unknown scheduler {opt_params.scheduler}")

        return [opt], [{
            'scheduler': sched,
            'interval': opt_params.sched_interval
        }]

    def log_all(self, out, phase="train"):
        on_step = (phase == "train")
        for k, v in out.items():
            self.log(f'{phase}/{k}', v, on_step=on_step, on_epoch=True)
        self.log(f"{phase}/accuracy", self.acc[f"{phase}_acc"], on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.loss(x, y, 'train')
        self.log_all(out, 'train')
        return out['loss']
    


