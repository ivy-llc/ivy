# global
from typing import Any, Dict, List

# local
from ...transformations.transformers.base_transformer import (
    BaseTransformer,
)
from ...transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...utils.import_utils import load_module_from_path

class BaseTranslatorConfig:
    _transformers: Dict[str, BaseTransformer] = {}

    _transformer_configs: Dict[str, Dict[BaseTransformer, BaseTransformerConfig]] = {}

    def __init__(
        self,
        data: Dict[str, Any],
        source: str = "torch",
        target: str = "torch_frontend",
        base_output_dir: str = "",
    ) -> None:
        self.source = source
        self.target = target
        self.base_output_dir = base_output_dir
        self.transformers: List[BaseTransformer] = []
        self.transformer_configs: Dict[str, BaseTransformerConfig] = {}
        self.profiling = False
        self.profiler = None
        # Load the standard objects to translate going from any frontend to ivy.
        # This dictionary contains a mapping of the module to search for the corresponding
        # list of standard objects (functions/methods) that we should translate
        # depending on the source set
        standard_objects_to_translate = {}
        standard_methods_to_translate = data.get("STANDARD_METHODS_TO_TRANSLATE", {})
        standard_functions_to_translate = data.get(
            "STANDARD_FUNCTIONS_TO_TRANSLATE", {}
        )

        # First, merge STANDARD_METHODS_TO_TRANSLATE
        for key, value in standard_methods_to_translate.items():
            # If the current source doesn't match the key, skip it
            if key != self.source:
                continue
            standard_objects_to_translate[key] = value

        # Now, merge STANDARD_FUNCTIONS_TO_TRANSLATE
        for key, value in standard_functions_to_translate.items():
            # If the current source doesn't match the key, skip it
            if key != self.source:
                continue
            if key in standard_objects_to_translate:
                # Combine the values if the key already exists
                for subkey, subvalue in value.items():
                    if subkey in standard_objects_to_translate[key]:
                        standard_objects_to_translate[key][subkey].extend(subvalue)
                    else:
                        standard_objects_to_translate[key][subkey] = subvalue
            else:
                # If the key doesn't exist, simply add it
                standard_objects_to_translate[key] = value

        # Optionally, remove duplicates from the lists (if needed)
        for top_key, nested_dict in standard_objects_to_translate.items():
            for nested_key in nested_dict:
                standard_objects_to_translate[top_key][nested_key] = list(
                    set(standard_objects_to_translate[top_key][nested_key])
                )

        self.standard_objects_to_translate: Dict[str, str] = (
            standard_objects_to_translate.get(self.source, {})
        )
