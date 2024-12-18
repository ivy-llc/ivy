# global
from typing import Dict

# local
from ..configurations.base_transformer_config import (
    BaseTransformerConfig,
)

from ...configs.transformer import native_torch_postprocessing_transformer_config_dev as native_torch_postprocessing_config


class NativeTorchCodePostProcessorConfig(BaseTransformerConfig):
    def __init__(self) -> None:
        super(NativeTorchCodePostProcessorConfig, self).__init__()

        data: Dict[str] = {
            key: value
            for key, value in native_torch_postprocessing_config.__dict__.items()
            if not key.startswith("__")
        }

        self.tensor_cls_map = data["TENSOR_ALIAS"]

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`torch` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        self.torch_meta = {
            "torch.__version__": torch.__version__,
        }
