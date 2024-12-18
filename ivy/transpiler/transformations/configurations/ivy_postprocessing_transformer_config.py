# global
from typing import Dict

# local
from ..configurations.base_transformer_config import (
    BaseTransformerConfig,
)

from ...configs.transformer import ivy_postprocessing_transformer_config_dev as ivy_postprocessing_config


class IvyCodePostProcessorConfig(BaseTransformerConfig):
    def __init__(self) -> None:
        super(IvyCodePostProcessorConfig, self).__init__()

        data: Dict[str] = {
            key: value
            for key, value in ivy_postprocessing_config.__dict__.items()
            if not key.startswith("__")
        }

        self.dtype_mapping = data["IVY_DTYPE_MAPPING"]
        self.default_dtype_mapping = data["IVY_DEFAULT_DTYPE_MAPPING"]
        self.ivy_cls_mapping = data["IVY_ARR_DTYPE_DEVICE"]
        self.native_cls_mapping = data["NATIVE_ARR_DTYPE_DEVICE"]
        self.hf_cls_mapping = data["HUGGINGFACE_CLS_MAPPING"]
        self.ivy_globs = data["IVY_GLOBS"]
