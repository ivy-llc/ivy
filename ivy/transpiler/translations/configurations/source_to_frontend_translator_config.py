# global
from typing import Dict, List

# local
from ...transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from ...transformations.transformers.deletion_transformer.base_transformer import (
    BaseNodeDeleter,
)
from ...transformations.transformers.base_transformer import (
    BaseTransformer,
)
from ...transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from ...transformations.transformers.globals_transformer.base_transformer import (
    BaseGlobalsTransformer,
)
from ...transformations.transformers.decorator_transformer.native_torch_decorator_transformer import (
    NativeTorchDecoratorRemover,
)
from ...transformations.transformers.typing_transformer.base_transformer import (
    BaseTypeHintRemover,
)
from ...transformations.transformers.closure_transformer.base_transformer import (
    BaseClosureToLocalTransformer,
)
from ...transformations.transformers.docstring_transformer.base_transformer import (
    BaseDocstringRemover,
)
from ...transformations.transformers.annotation_transformer.base_transformer import (
    BaseTypeAnnotationRemover,
)
from ...transformations.transformers.recursive_transformer.native_torch_recursive_transformer import (
    NativeTorchRecurser,
)
from ...transformations.transformers.postprocessing_transformer.native_torch_postprocessing_transformer import (
    NativeTorchCodePostProcessor,
)
from ...transformations.configurations.native_torch_postprocessing_config import (
    NativeTorchCodePostProcessorConfig,
)
from ...transformations.transformers.method_transformer.native_torch_method_transformer import (
    NativeTorchMethodToFunctionConverter,
)
from ...transformations.transformers.profiling_transformer.base_transformer import (
    BaseProfilingTransformer,
)
from ...configs.translator import source_to_frontend_translator_config_dev as source_to_frontend_translator_config


class SourceToFrontendTranslatorConfig(BaseTranslatorConfig):
    """Holds the configurations to run the SourceToFrontendTranslator."""

    _native_transformers: Dict[str, BaseTransformer] = {
        "torch": {
            "decorator_remover": NativeTorchDecoratorRemover,
            "recurser": NativeTorchRecurser,
            "postprocessing": NativeTorchCodePostProcessor,
        }
    }

    _native_transformer_configs: Dict[
        str, Dict[BaseTransformer, BaseTransformerConfig]
    ] = {"torch": {NativeTorchCodePostProcessor: NativeTorchCodePostProcessorConfig}}

    def __init__(
        self,
        source="torch",
        target="torch_frontend",
        base_output_dir="",
    ) -> None:
        data = {
            key: value
            for key, value in source_to_frontend_translator_config.__dict__.items()
            if not key.startswith("__")
        }

        super(SourceToFrontendTranslatorConfig, self).__init__(
            data=data, source=source, target=target, base_output_dir=base_output_dir
        )

        # Load the transformations necessary for this translator
        self.transformers: List[BaseTransformer] = [
            BaseNodeDeleter,
            BaseTypeHintRemover,
            BaseDocstringRemover,
            BaseTypeAnnotationRemover,
            self._native_transformers[source]["decorator_remover"],
            NativeTorchMethodToFunctionConverter,
            BaseNameCanonicalizer,
            BaseGlobalsTransformer,
            self._native_transformers[source]["recurser"],
            BaseClosureToLocalTransformer,
            self._native_transformers[source]["postprocessing"],
        ]

        self.profiler = None

        # Load the configurations for the transformers as well
        self.transformer_configs: List[BaseTransformerConfig] = [
            (
                self._native_transformer_configs[source][transformer]
                if transformer in self._native_transformer_configs[source]
                else BaseTransformerConfig
            )
            for transformer in self.transformers
        ]
