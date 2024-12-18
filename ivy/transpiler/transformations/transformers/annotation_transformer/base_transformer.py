# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ...transformers.base_transformer import (
    BaseTransformer,
)


class BaseTypeAnnotationRemover(BaseTransformer):
    """
    A class to remove type annotations in gast.FunctionDef and gast.ClassDef.
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
        self._remove_type_annotations(node)
        self.generic_visit(node)
        return node

    def _remove_type_annotations(self, node):
        # Remove the return annotation
        node.returns = None

        # Remove argument type annotations
        for arg in node.args.args:
            arg.annotation = None

        # Remove positional-only argument type annotations
        for arg in node.args.posonlyargs:
            arg.annotation = None

        # Remove positional-only argument type annotations
        for arg in node.args.kwonlyargs:
            arg.annotation = None

        # Remove vararg and kwarg type annotations
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
