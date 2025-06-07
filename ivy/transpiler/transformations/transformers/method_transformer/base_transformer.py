from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.base_transformer import (
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
