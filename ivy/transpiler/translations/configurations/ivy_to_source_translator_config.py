# global
from typing import Dict, List

# local
from ivy.transpiler.translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from ivy.transpiler.transformations.transformers.base_transformer import (
    BaseTransformer,
)
from ivy.transpiler.transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from ivy.transpiler.transformations.transformers.globals_transformer.base_transformer import (
    BaseGlobalsTransformer,
)
from ivy.transpiler.transformations.transformers.deletion_transformer.ivy_deletion_transformer import (
    IvyNodeDeleter,
)
from ivy.transpiler.transformations.transformers.decorator_transformer.ivy_decorator_transformer import (
    IvyDecoratorRemover,
)
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformers.docstring_transformer.base_transformer import (
    BaseDocstringRemover,
)
from ivy.transpiler.transformations.configurations.ivy_recursive_transformer_config import (
    IvyRecurserConfig,
)
from ivy.transpiler.transformations.configurations.ivy_postprocessing_transformer_config import (
    IvyCodePostProcessorConfig,
)
from ivy.transpiler.transformations.transformers.method_transformer.ivy_method_transformer import (
    IvyMethodToFunctionConverter,
)
from ivy.transpiler.transformations.transformers.dunders_transformer.base_transformer import (
    BaseDundersTransformer,
)
from ivy.transpiler.transformations.transformers.preprocessing_transformer.ivy_preprocessing_transformer import (
    IvyCodePreProcessor,
)
from ivy.transpiler.transformations.transformers.postprocessing_transformer.ivy_to_tf_postprocessing_transformer import (
    IvyToTFCodePostProcessor,
)
from ivy.transpiler.transformations.transformers.postprocessing_transformer.ivy_to_jax_postprocessing_transformer import (
    IvyToJAXCodePostProcessor,
)
from ivy.transpiler.transformations.transformers.postprocessing_transformer.ivy_to_numpy_postprocessing_transformer import (
    IvyToNumpyCodePostProcessor,
)
from ivy.transpiler.transformations.transformers.recursive_transformer.ivy_recursive_transformer import (
    IvyRecurser,
)
from ivy.transpiler.transformations.transformers.native_layers_transformer.ivy_to_tf_native_layer_transformer import (
    PytorchToKerasLayer,
)
from ivy.transpiler.transformations.transformers.native_layers_transformer.ivy_to_jax_native_layer_transformer import (
    PytorchToFlaxLayer,
)
from ivy.transpiler.transformations.transformers.miscellaneous_transformers.hf_flax_transformer import (
    HFPretrainedFlaxTransformer,
)
import ivy.transpiler.configs.translator.ivy_to_source_translator_config_dev as ivy_to_source_translator_config


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
