import os
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass, field

import yaml
from omegaconf import OmegaConf, DictConfig

__all__ = ["FDConfig", "ConfigType", "load_config", "print_config",
           "ModelConfig", "OptimizerConfig", "DataConfig", "TrainerConfig",
           "MiscConfig", "PLTrainerConfig", "LossConfig", "SAMConfig"]


@dataclass
class ModelConfig:
    model_cls: str = "WideResnet"
    num_inputs: int = 3
    # Dimension of the output of the model (ie number of classes for a classification problem).
    num_outputs: int = 10
    # How many resnet blocks to add to each group (should be 4 blocks for a WRN28, and 6 for a WRN40).
    wrn_blocks: int = 4
    # The multiplier to apply to the number of filters in the model (1 is classical resnet, 10 for WRN28-10, etc...).
    wrn_multiplier: int = 10
    wrn_use_additional_skips: bool = False
    mobile_net_small: bool = True
    mobile_net_pretrained: bool = True


@dataclass
class OptimizerConfig:
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    wd: float = 5e-4
    nesterov: bool = True
    # "cosine" or None
    scheduler: Optional[str] = "cosine"
    # "epoch" or "step" (Should be step for both cosine to match original jax implementation)
    sched_interval: str = "step"
    warmup_epochs: int = 0
    clip_grad_norm: Optional[float] = 5.0


@dataclass
class DataConfig:
    data_root: str = str(Path("~/data").expanduser())
    dataset_cls: str = "ClassificationDataset"
    dataset: str = "cifar10"
    train_batch_size: int = 256
    val_batch_size: int = 256
    test_batch_size: int = 256
    train_workers: int = 4
    val_workers: int = 4
    test_workers: int = 4
    # Cutout for CIFAR10
    use_cutout: bool = True
    # Autoaugment for CIFAR10
    use_autoaugment: bool = True


@dataclass
class LossConfig:
    label_smoothing: Optional[float] = None


@dataclass
class PLTrainerConfig:
    accerelator: str = "gpu"
    devices: int = 1
    cudnn_benchmark: bool = True
    log_freq: int = 10
    monitor: str = "val/accuracy"
    monitor_decreasing: bool = False
    precision: int = 32


@dataclass
class TrainerConfig:
    trainer: str = "SAMTrainer"
    pl: PLTrainerConfig = PLTrainerConfig()
    max_epochs: int = 200
    out_dir: str = str(Path("~/outputs/sam").expanduser())


@dataclass
class MiscConfig:
    seed: int = 42


@dataclass
class SAMConfig:
    use_sam: bool = True
    adaptive: bool = False
    rho: float = 0.05


@dataclass
class FDConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    misc: MiscConfig = MiscConfig()
    sam: SAMConfig = SAMConfig()
    project: str = "r252_sam"
    group: str = "default"


ConfigType = Union[FDConfig, DictConfig]


def load_config(yaml_path: Union[str, bytes, os.PathLike]) -> ConfigType:
    schema = OmegaConf.structured(FDConfig)
    conf = OmegaConf.load(yaml_path)
    return OmegaConf.merge(schema, conf)


def print_config(config: ConfigType):
    cont = OmegaConf.to_container(config)
    print(yaml.dump(cont, allow_unicode=True, default_flow_style=False), end='')
