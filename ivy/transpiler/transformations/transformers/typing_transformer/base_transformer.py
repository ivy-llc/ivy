# global
import gast

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)


class BaseTypeHintRemover(BaseTransformer):
    """
    A class to remove all the type hints in a gast AST.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        # Replace the annotated assignment with a simple assignment to None
        new_node = gast.AnnAssign(
            target=node.target,
            annotation=gast.parse("typing.Any").body[0].value,
            value=node.value,
            simple=node.simple,
        )
        self.transformer._imports.add(("typing", None))
        return new_node
