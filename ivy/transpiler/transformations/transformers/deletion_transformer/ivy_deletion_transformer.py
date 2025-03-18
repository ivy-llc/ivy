# local
import gast
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)
from ....utils.ast_utils import ast_to_source_code


class IvyNodeDeleter(BaseTransformer):
    """
    A class to delete certain nodes from the AST based on some hard-coded heuristics/patterns.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        self.visit(self.root)

    def visit_Expr(self, node):
        if self._is_check_jax_x64_flag_call(node.value):
            return None  # Delete the node
        elif self._is_array_mode_call(node.value):
            return None  # Delete the node
        else:
            return self.generic_visit(node)

    def _is_check_jax_x64_flag_call(self, node):
        return (
            isinstance(node, gast.Call)
            and ast_to_source_code(node.func).strip()
            == "ivy.utils.assertions._check_jax_x64_flag"
        )

    def _is_array_mode_call(self, node):
        return isinstance(node, gast.Call) and ast_to_source_code(
            node.func
        ).strip() in ("ivy.set_array_mode", "ivy.unset_array_mode")
