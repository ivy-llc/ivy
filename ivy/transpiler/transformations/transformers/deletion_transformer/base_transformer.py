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


class BaseNodeDeleter(BaseTransformer):
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

    def visit_Module(self, node):
        self._delete_matching_if_nodes(node)
        self._delete_use_checkpointing(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if node.name == "use_checkpointing":
            return None  # Delete the node
        else:
            self._delete_matching_if_nodes(node)
            self._delete_log_api_usage_once(node)
            self._delete_post_init(node)
            self.generic_visit(node)
            return node

    def visit_Call(self, node):
        return self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)
        new_body = []
        for stmt in node.body:
            if self._is_checkpoint_call(stmt):
                # Replace the checkpoint call with 'pass'
                stmt = gast.Pass()
            elif self._is_matching_warn_if_padding_and_no_attention_mask_call(stmt):
                continue  # skip this statement
            else:
                stmt = stmt
            new_body.append(self.generic_visit(stmt))
        node.body = new_body
        return node

    def visit_With(self, node):
        if self._is_no_grad_context_manager(
            node
        ) or self._is_enable_grad_context_manager(node):
            return node.body
        elif self._is_autocast_context_manager(node):
            return node.body
        elif self._is_inference_mode_context_manager(node):
            return node.body
        else:
            return self.generic_visit(node)

    def _delete_matching_if_nodes(self, node):
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, gast.If) and self._is_matching_if_node(stmt):
                continue  # Skip this statement
            new_body.append(stmt)
        node.body = new_body

    def _delete_use_checkpointing(self, node):
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, gast.FunctionDef) and stmt.name == "use_checkpointing":
                continue  # Skip this statement
            new_body.append(stmt)
        node.body = new_body

    def _delete_log_api_usage_once(self, node):
        new_body = []
        for stmt in node.body:
            if isinstance(
                stmt, gast.Expr
            ) and self._is_matching_log_api_usage_once_call(stmt.value):
                continue  # Skip this statement
            new_body.append(stmt)
        node.body = new_body

    def _delete_post_init(self, node):
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, gast.Expr) and self._is_matching_post_init_call(
                stmt.value
            ):
                continue  # Skip this statement
            new_body.append(stmt)
        node.body = new_body

    def _is_checkpoint_call(self, node):
        return (
            isinstance(node, gast.Assign)
            and isinstance(node.value, gast.Call)
            and isinstance(node.value.func, gast.Attribute)
            and node.value.func.attr == "checkpoint"
        )

    def _is_matching_if_node(self, node):
        return (
            isinstance(node.test, gast.Call)
            and isinstance(node.test.func, gast.Attribute)
            and node.test.func.attr == "has_torch_function_variadic"
            and isinstance(node.test.func.value, gast.Attribute)
            and node.test.func.value.attr == "overrides"
            and isinstance(node.test.func.value.value, gast.Name)
            and node.test.func.value.value.id == "torch"
        )

    def _is_matching_annotate_call(self, node):
        func_str = ast_to_source_code(node)
        return "torch.jit.annotate" in func_str

    # is a HF (pytorch) PreTrainedModel-specific method but not a method of TFPreTrainedModel
    # TODO: maybe we should translate self.post_init() instead of deleting it.
    def _is_matching_post_init_call(self, node):
        func_str = ast_to_source_code(node).strip()
        return func_str == "self.post_init()"

    # is a HF (pytorch) PreTrainedClass-specific method
    def _is_matching_warn_if_padding_and_no_attention_mask_call(self, node):
        func_str = ast_to_source_code(node)
        return "warn_if_padding_and_no_attention_mask" in func_str

    def _is_no_grad_context_manager(self, node):
        return (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, gast.Call)
            and isinstance(node.items[0].context_expr.func, gast.Attribute)
            and node.items[0].context_expr.func.attr == "no_grad"
            and isinstance(node.items[0].context_expr.func.value, gast.Name)
            and node.items[0].context_expr.func.value.id == "torch"
        )

    def _is_enable_grad_context_manager(self, node):
        return (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, gast.Call)
            and isinstance(node.items[0].context_expr.func, gast.Attribute)
            and node.items[0].context_expr.func.attr == "enable_grad"
            and isinstance(node.items[0].context_expr.func.value, gast.Name)
            and node.items[0].context_expr.func.value.id == "torch"
        )

    def _is_autocast_context_manager(self, node):
        return (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, gast.Call)
            and isinstance(node.items[0].context_expr.func, gast.Attribute)
            and node.items[0].context_expr.func.attr == "autocast"
            and isinstance(node.items[0].context_expr.func.value, gast.Name)
            and node.items[0].context_expr.func.value.id == "torch"
        )

    def _is_inference_mode_context_manager(self, node):
        return (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, gast.Call)
            and isinstance(node.items[0].context_expr.func, gast.Attribute)
            and node.items[0].context_expr.func.attr == "inference_mode"
            and isinstance(node.items[0].context_expr.func.value, gast.Name)
            and node.items[0].context_expr.func.value.id == "torch"
        )

    def _is_matching_log_api_usage_once_call(self, node):
        return (
            isinstance(node, gast.Call)
            and isinstance(node.func, gast.Name)
            and node.func.id == "_log_api_usage_once"
            and isinstance(node.args[0], gast.Name)
            and node.args[0].id == "self"
        )
