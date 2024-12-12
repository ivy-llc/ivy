# local
from source_to_source_translator.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from source_to_source_translator.transformations.transformer import Transformer
from source_to_source_translator.transformations.transformers.base_transformer import (
    BaseTransformer,
)


class BaseMethodToFunctionConverter(BaseTransformer):
    """
    A class to convert method calls to function calls in a gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        pass
