# local
from source_to_source_translator.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from source_to_source_translator.transformations.transformer import Transformer
from source_to_source_translator.transformations.transformers.base_transformer import (
    BaseTransformer,
)


class BaseCodePostProcessor(BaseTransformer):
    """
    A class to post process the given gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        new_name="",
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.new_name = new_name

    def transform(self):
        pass
