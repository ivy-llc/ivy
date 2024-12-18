# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
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
