# global
from typing import Dict, List

# local
from transpiler.translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from transpiler.transformations.transformers.base_transformer import (
    BaseTransformer,
)
from transpiler.transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from transpiler.transformations.transformers.globals_transformer.base_transformer import (
    BaseGlobalsTransformer,
)
from transpiler.transformations.transformers.deletion_transformer.ivy_deletion_transformer import (
    IvyNodeDeleter,
)
from transpiler.transformations.transformers.decorator_transformer.ivy_decorator_transformer import (
    IvyDecoratorRemover,
)
from transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from transpiler.transformations.transformers.docstring_transformer.base_transformer import (
    BaseDocstringRemover,
)
from transpiler.transformations.configurations.ivy_recursive_transformer_config import (
    IvyRecurserConfig,
)
from transpiler.transformations.configurations.ivy_postprocessing_transformer_config import (
    IvyCodePostProcessorConfig,
)
from transpiler.transformations.transformers.method_transformer.ivy_method_transformer import (
    IvyMethodToFunctionConverter,
)
from transpiler.transformations.transformers.dunders_transformer.base_transformer import (
    BaseDundersTransformer,
)
from transpiler.transformations.transformers.preprocessing_transformer.ivy_preprocessing_transformer import (
    IvyCodePreProcessor,
)
from transpiler.transformations.transformers.postprocessing_transformer.ivy_to_tf_postprocessing_transformer import (
    IvyToTFCodePostProcessor,
)
from transpiler.transformations.transformers.postprocessing_transformer.ivy_to_jax_postprocessing_transformer import (
    IvyToJAXCodePostProcessor,
)
from transpiler.transformations.transformers.postprocessing_transformer.ivy_to_numpy_postprocessing_transformer import (
    IvyToNumpyCodePostProcessor,
)
from transpiler.transformations.transformers.recursive_transformer.ivy_recursive_transformer import (
    IvyRecurser,
)
from transpiler.transformations.transformers.native_layers_transformer.ivy_to_tf_native_layer_transformer import (
    PytorchToKerasLayer,
)
from transpiler.transformations.transformers.native_layers_transformer.ivy_to_jax_native_layer_transformer import (
    PytorchToFlaxLayer,
)
from transpiler.transformations.transformers.miscellaneous_transformers.hf_flax_transformer import (
    HFPretrainedFlaxTransformer,
)
import transpiler.configs.translator.ivy_to_source_translator_config_dev as ivy_to_source_translator_config


class IvyToSourceTranslatorConfig(BaseTranslatorConfig):
    """Holds the configurations to run the IvyToSourceTranslator."""

    _ivy_transformers: Dict[str, BaseTransformer] = {}

    _ivy_transformer_configs: Dict[BaseTransformer, BaseTransformerConfig] = {
        IvyRecurser: IvyRecurserConfig,
        IvyToTFCodePostProcessor: IvyCodePostProcessorConfig,
        IvyToJAXCodePostProcessor: IvyCodePostProcessorConfig,
        IvyToNumpyCodePostProcessor: IvyCodePostProcessorConfig,
    }

    def __init__(self, source="ivy", target="tensorflow", base_output_dir="") -> None:
        data = {
            key: value
            for key, value in ivy_to_source_translator_config.__dict__.items()
            if not key.startswith("__")
        }

        super(IvyToSourceTranslatorConfig, self).__init__(
            data=data, source=source, target=target, base_output_dir=base_output_dir
        )

        # Load the transformations necessary for this translator
        if target == "tensorflow":
            self.transformers: List[BaseTransformer] = [
                IvyNodeDeleter,
                IvyDecoratorRemover,
                # BaseTypeHintRemover,
                BaseDocstringRemover,
                # BaseTypeAnnotationRemover,
                IvyMethodToFunctionConverter,
                BaseDundersTransformer,
                IvyCodePreProcessor,
                BaseNameCanonicalizer,
                BaseGlobalsTransformer,
                IvyRecurser,
                IvyToTFCodePostProcessor,
                PytorchToKerasLayer,
            ]
        elif target == "jax":
            self.transformers: List[BaseTransformer] = [
                IvyNodeDeleter,
                IvyDecoratorRemover,
                # BaseTypeHintRemover,
                BaseDocstringRemover,
                # BaseTypeAnnotationRemover,
                IvyMethodToFunctionConverter,
                BaseDundersTransformer,
                IvyCodePreProcessor,
                BaseNameCanonicalizer,
                BaseGlobalsTransformer,
                IvyRecurser,
                IvyToJAXCodePostProcessor,
                PytorchToFlaxLayer,
                HFPretrainedFlaxTransformer,
            ]
        elif target == "numpy":
            self.transformers: List[BaseTransformer] = [
                IvyNodeDeleter,
                IvyDecoratorRemover,
                # BaseTypeHintRemover,
                BaseDocstringRemover,
                # BaseTypeAnnotationRemover,
                IvyMethodToFunctionConverter,
                BaseDundersTransformer,
                IvyCodePreProcessor,
                BaseNameCanonicalizer,
                BaseGlobalsTransformer,
                IvyRecurser,
                IvyToNumpyCodePostProcessor,
            ]
        self.profiler = None

        # Load the configurations for the transformers as well
        self.transformer_configs: List[BaseTransformerConfig] = [
            (
                self._ivy_transformer_configs[transformer]
                if transformer in self._ivy_transformer_configs
                else BaseTransformerConfig
            )
            for transformer in self.transformers
        ]
