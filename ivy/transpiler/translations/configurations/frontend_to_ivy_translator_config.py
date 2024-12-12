# global
from typing import Dict, List

# local
from transpiler.translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from transpiler.transformations.transformers.base_transformer import (
    BaseTransformer,
)
from transpiler.transformations.transformers.annotation_transformer.base_transformer import (
    BaseTypeAnnotationRemover,
)
from transpiler.transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from transpiler.transformations.transformers.globals_transformer.base_transformer import (
    BaseGlobalsTransformer,
)
from transpiler.transformations.transformers.decorator_transformer.frontend_torch_decorator_transformer import (
    FrontendTorchDecoratorRemover,
)
from transpiler.transformations.transformers.inject_transformer.base_transformer import (
    BaseSuperMethodsInjector,
)
from transpiler.transformations.transformers.method_transformer.frontend_torch_method_transformer import (
    FrontendTorchMethodToFunctionConverter,
)
from transpiler.transformations.transformers.postprocessing_transformer.frontend_torch_postprocessing_transformer import (
    FrontendTorchCodePostProcessor,
)
from transpiler.transformations.transformers.preprocessing_transformer.frontend_torch_preprocessing_transformer import (
    FrontendTorchCodePreProcessor,
)
from transpiler.transformations.transformers.recursive_transformer.frontend_torch_recursive_transformer import (
    FrontendTorchRecurser,
)
from transpiler.transformations.transformers.typing_transformer.base_transformer import (
    BaseTypeHintRemover,
)
from transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from transpiler.transformations.configurations.frontend_torch_postprocessing_transformer_config import (
    FrontendTorchCodePostProcessorConfig,
)
from transpiler.transformations.transformers.inplace_update_transformer.base_transformer import (
    BaseInplaceUpdateTransformer,
)

from transpiler.transformations.transformers.profiling_transformer.base_transformer import (
    BaseProfilingTransformer,
)
import transpiler.configs.translator.frontend_to_ivy_translator_config_dev as frontend_to_ivy_translator_config


class FrontendToIvyTranslatorConfig(BaseTranslatorConfig):
    """Holds the configurations to run the FrontendToIvyTranslator."""

    _frontend_transformers: Dict[str, Dict[str, BaseTransformer]] = {
        "torch_frontend": {
            "decorator_remover": FrontendTorchDecoratorRemover,
            "preprocessor": FrontendTorchCodePreProcessor,
            "converter": FrontendTorchMethodToFunctionConverter,
            "recurser": FrontendTorchRecurser,
            "postprocessor": FrontendTorchCodePostProcessor,
        }
    }

    _frontend_transformer_configs: Dict[
        str, Dict[BaseTransformer, BaseTransformerConfig]
    ] = {
        "torch_frontend": {
            FrontendTorchCodePostProcessor: FrontendTorchCodePostProcessorConfig,
        }
    }

    def __init__(
        self, source="torch_frontend", target="ivy", base_output_dir=""
    ) -> None:
        data = {
            key: value
            for key, value in frontend_to_ivy_translator_config.__dict__.items()
            if not key.startswith("__")
        }

        super(FrontendToIvyTranslatorConfig, self).__init__(
            data=data, source=source, target=target, base_output_dir=base_output_dir
        )

        # Load the transformations necessary for this translator
        self.transformers: List[BaseTransformer] = [
            BaseSuperMethodsInjector,
            self._frontend_transformers[source]["decorator_remover"],
            BaseTypeHintRemover,
            BaseTypeAnnotationRemover,
            self._frontend_transformers[source]["preprocessor"],
            self._frontend_transformers[source]["converter"],
            BaseNameCanonicalizer,
            BaseGlobalsTransformer,
            self._frontend_transformers[source]["recurser"],
            self._frontend_transformers[source]["postprocessor"],
            BaseInplaceUpdateTransformer,
        ]

        self.profiler = None

        # Load the configurations for the transformers as well
        self.transformer_configs: List[BaseTransformerConfig] = [
            (
                self._frontend_transformer_configs[source][transformer]
                if transformer in self._frontend_transformer_configs[source]
                else BaseTransformerConfig
            )
            for transformer in self.transformers
        ]
