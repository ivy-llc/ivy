# global
from typing import Dict

# local
from ..configurations.base_transformer_config import (
    BaseTransformerConfig,
)

from ...configs.transformer import frontend_torch_postprocessing_transformer_config_dev as frontend_torch_postprocessing_config


class FrontendTorchCodePostProcessorConfig(BaseTransformerConfig):
    def __init__(self) -> None:
        super(FrontendTorchCodePostProcessorConfig, self).__init__()

        data: Dict[str] = {
            key: value
            for key, value in frontend_torch_postprocessing_config.__dict__.items()
            if not key.startswith("__")
        }

        self.axis_map = data["AXIS_MAP"]
        self.dtype_mapping = data["DTYPE_MAPPING"]
        self.array_and_module_map = data["ARRAY_AND_MODULE_MAPPING"]
