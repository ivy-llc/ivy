# global
import gast

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import (
    ast_to_source_code,
    is_unpacking_assignment,
    replace_placeholders,
)
from .base_transformer import (
    BaseCodePreProcessor,
)


def filter_statements(statements):
    return [
        stmt
        for stmt in statements
        if not (
            (
                isinstance(stmt, gast.Expr)
                and isinstance(stmt.value, gast.Call)
                and ast_to_source_code(stmt.value.func).strip()
                in ("_check_inplace_update_support",)
            )
        )
    ]


class IvyCodePreProcessor(BaseCodePreProcessor):
    """
    A class to perform preprocessing on a given ivy gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        new_name="tensor",
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.new_name = new_name

    def transform(self):
        self.visit(self.root)

    def visit_Try(self, node):
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        node = self._maybe_replace_conversion_nodes_in_assignments(node)
        node = self._maybe_remove_shape_attribute_access_in_assignments(node)

        # Check for unpacking assingments and transform them to use explicit indexing notations
        # Needed to work with tf.function
        if is_unpacking_assignment(node):
            left_targets = node.targets[0].elts
            right_value = node.value

            indexed_values = []
            for i, target in enumerate(left_targets):
                indexed_value = self._create_indexed_value(target, i, right_value)
                indexed_values.append(indexed_value)

            # Create a new assignment node with the modified right-hand side
            new_assign_node = gast.Assign(
                targets=node.targets,
                value=gast.Tuple(elts=indexed_values, ctx=gast.Load()),
            )
            node = gast.copy_location(new_assign_node, node)

        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)

        # Filter out unwanted function calls in the body
        node.body = filter_statements(node.body)

        return node

    def visit_If(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)

        # Filter out unwanted function calls in the body and orelse
        node.body = filter_statements(node.body)
        node.orelse = filter_statements(node.orelse)

        # Filter out spcific branches e.g. `ivy.is_ivy_array` in
        # inplace_update (tf backend)
        # TODO: the if ivy.is_ivy_array branch contains logic specific
        # to view-handling. Hence, we should not filter it out
        node = self._maybe_remove_elif_branches(node)

        return node

    def _create_indexed_value(self, value, index, right_value):
        # Recursive function to create indexed value with nested subscripts
        if isinstance(value, gast.Name):
            return gast.Subscript(
                value=right_value,
                slice=gast.Constant(value=index, kind=None),
                ctx=gast.Load(),
            )
        elif isinstance(value, gast.Attribute):
            return gast.Subscript(
                value=right_value,
                slice=gast.Constant(value=index, kind=None),
                ctx=gast.Load(),
            )
        elif isinstance(value, gast.Subscript):
            return gast.Subscript(
                value=self._create_indexed_value(right_value.value, index),
                slice=right_value.slice,
                ctx=gast.Load(),
            )
        elif isinstance(value, gast.Tuple):
            indexed_elts = []
            for idx, elt in enumerate(value.elts):
                subscript = self._create_indexed_value(elt, index, right_value)
                subscript = gast.Subscript(
                    value=subscript, slice=gast.Constant(value=idx, kind=None)
                )
                indexed_elts.append(subscript)
            return gast.Tuple(elts=indexed_elts, ctx=gast.Load())

    def _maybe_remove_elif_branches(self, node):
        # Check if the original if statement has an "elif" part
        # which we'd want to remove.
        # TODO: Make this logic work with arbitrary if-elif-else nodes
        if len(node.orelse) == 1 and isinstance(node.orelse[0], gast.If):
            elif_node = node.orelse[0]

            # Check the condition for the elif node
            if any(
                condition in ast_to_source_code(elif_node.test).strip()
                for condition in ("ivy.is_ivy_array",)
            ):
                node.orelse = elif_node.orelse

        return node

    # NOTE: this can potentially be removed if we later on decide to translate `ivy.Shape`
    # right now, we statically replace `ivy.Shape` --> tuple.
    # The check here is meant to do this : _check_bounds_and_get_shape(..).shape --> _check_bounds_and_get_shape(..)
    def _maybe_remove_shape_attribute_access_in_assignments(
        self, node, attrs_to_check_for=("shape",)
    ):
        # for certain nodes we might want to remove attribute access on
        if (
            isinstance(node.value, gast.Attribute)
            and "_check_bounds_and_get_shape"
            in ast_to_source_code(node.value.value).strip()
            and node.value.attr in attrs_to_check_for
        ):
            node.value = node.value.value
        return node

    def _maybe_replace_conversion_nodes_in_assignments(self, node):
        # for nodes like `ivy.to_ivy`, `ivy.args_to_native`, `ivy.to_native` etc.
        # we modify them so that rather than a function call, we just return return the
        # arguments as is.
        def __generate_rhs_for_tuple_node(node: gast.Tuple):
            new_elts = []
            for elt in node.elts:
                if isinstance(elt, gast.Tuple):
                    new_elts.append(__generate_rhs_for_tuple_node(elt))
                else:
                    new_elts.append(
                        gast.Expr(
                            value=gast.Name(
                                id="_",
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            )
                        )
                    )
            return gast.Tuple(elts=new_elts, ctx=gast.Load())

        def __flatten_nested_tuple_nodes_elts(nodes):
            elts = []
            for node in nodes:
                if isinstance(node, gast.Tuple):
                    elts.extend(node.elts)
                else:
                    elts.append(node)
            return elts

        if isinstance(node.value, gast.Call) and any(
            m in ast_to_source_code(node.value.func)
            for m in ("to_ivy", "args_to_native", "to_native")
        ):
            call_node = node.value
            if len(call_node.args) > 1:
                # case1:  `x,y = ivy.args_to_native(x, y)` --> `x, y= x,y`
                new_rhs_nodes = []
                for target in node.targets:
                    if isinstance(target, gast.Tuple):
                        new_rhs_nodes.append(__generate_rhs_for_tuple_node(target))
                    else:
                        new_rhs_nodes.append(
                            gast.Expr(
                                value=gast.Name(
                                    id="_",
                                    ctx=gast.Load(),
                                    annotation=None,
                                    type_comment=None,
                                )
                            )
                        )
                new_rhs_nodes = __flatten_nested_tuple_nodes_elts(new_rhs_nodes)
                new_rhs = gast.Tuple(
                    elts=new_rhs_nodes,
                    ctx=gast.Load(),
                )
                new_rhs = gast.fix_missing_locations(new_rhs)
                new_rhs = replace_placeholders(new_rhs, call_node.args)
                node.value = new_rhs
            elif len(call_node.args) > 0:
                # case2:  `x = ivy.to_ivy(x)` --> `x= x`
                node.value = call_node.args[0]
            else:
                # case3: `x = ivy.astype(x, dtype).to_native()` --> `x= ivy.astype(x, dtype)`
                node.value = call_node.func.value
        return node
