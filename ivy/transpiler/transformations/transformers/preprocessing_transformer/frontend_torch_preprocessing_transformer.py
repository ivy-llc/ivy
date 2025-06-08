from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.preprocessing_transformer.base_transformer import (
    BaseCodePreProcessor,
)


class FrontendTorchCodePreProcessor(BaseCodePreProcessor):
    """
    A class to perform preprocessing on a given frontend gast AST.
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
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)
        return node
