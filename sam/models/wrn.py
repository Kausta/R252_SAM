import sam.nn as snn
from sam.util import ConfigType

__all__ = ["WideResnet"]

class WideResnet(snn.WideResnet):
    def __init__(self, config: ConfigType):
        super().__init__(
            config.model.wrn_blocks,
            config.model.wrn_multiplier,
            config.model.num_inputs,
            config.model.num_outputs,
            use_additional_skips=config.model.wrn_use_additional_skips
        )