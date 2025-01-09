# global
from typing import Dict

# local
from ..configurations.base_transformer_config import (
    BaseTransformerConfig,
)

from ...configs.transformer import ivy_recursive_transformer_config_dev as ivy_recursive_config


class IvyRecurserConfig(BaseTransformerConfig):
    def __init__(self) -> None:
        super(IvyRecurserConfig, self).__init__()

        data: Dict[str] = {
            key: value
            for key, value in ivy_recursive_config.__dict__.items()
            if not key.startswith("__")
        }

        self.curr_backend_call_regex = data["CURR_BACKEND_CALL_REGEX_PATTERN"]
