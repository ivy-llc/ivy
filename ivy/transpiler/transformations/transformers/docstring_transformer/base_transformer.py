# local
import gast
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)


class BaseDocstringRemover(BaseTransformer):
    """
    A class to remove docstrings in gast.FunctionDef and gast.ClassDef,
    except when the docstring is the only thing in the body.
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
        self._remove_docstring_if_needed(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self._remove_docstring_if_needed(node)
        self.generic_visit(node)
        return node

    def _remove_docstring_if_needed(self, node):
        # Remove the return annotation
        node.returns = None

        # Check if the body contains only the docstring and nothing else
        if len(node.body) == 1 and self._is_docstring_node(node.body[0]):
            # The docstring is the only thing in the body, so do not remove it
            return

        # Remove the docstring if it exists and there are other elements in the body
        if len(node.body) > 0 and self._is_docstring_node(node.body[0]):
            node.body = node.body[1:]  # Remove the docstring

        # Remove the __doc__ attribute if it exists
        node.body = [stmt for stmt in node.body if not self._is_doc_attribute(stmt)]

    def _is_docstring_node(self, node):
        """
        Check if a given node is a docstring node (a string literal at the start of a function/class).
        """
        return (
            isinstance(node, gast.Expr)
            and isinstance(node.value, gast.Constant)
            and isinstance(node.value.value, str)
        )

    def _is_doc_attribute(self, stmt):
        """
        Check if a given statement is an assignment to __doc__.
        """
        return (
            isinstance(stmt, gast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], gast.Name)
            and stmt.targets[0].id == "__doc__"
        )
