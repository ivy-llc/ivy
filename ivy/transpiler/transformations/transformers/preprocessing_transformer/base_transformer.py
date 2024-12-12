# local
from transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from transpiler.transformations.transformer import Transformer
from transpiler.transformations.transformers.base_transformer import (
    BaseTransformer,
)


class BaseCodePreProcessor(BaseTransformer):
    """
    A class to preprocess the given gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        new_name="arr",
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.new_name = new_name

    def transform(self):
        pass
