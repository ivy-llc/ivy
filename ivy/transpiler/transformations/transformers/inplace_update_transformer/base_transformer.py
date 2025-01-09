# global
import copy
import gast
import re

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)
from ....utils.api_utils import (
    TRANSLATED_OBJ_PREFIX,
    TRANSLATED_OBJ_SUFFIX,
)
from ....utils.ast_utils import ast_to_source_code, set_parents


def clean_function_name(func_name):
    """
    Remove any prefixes and suffixes from the function name according to the TRANSLATED_OBJ_PREFIX
    and TRANSLATED_OBJ_SUFFIX lists, then return the cleaned function name for regex testing.
    This only removes **one** leading and trailing "_" after stripping any prefix/suffix.
    """
    prefix_removed, suffix_removed = False, False

    # Remove any matching prefix
    for prefix in TRANSLATED_OBJ_PREFIX:
        if func_name.startswith(prefix):
            func_name = func_name[len(prefix) :]
            prefix_removed = prefix_removed if prefix_removed else True
            break  # Only remove the first matching prefix

    # Remove any matching suffix
    for suffix in TRANSLATED_OBJ_SUFFIX:
        if func_name.endswith(suffix):
            func_name = func_name[: -len(suffix)]
            suffix_removed = suffix_removed if suffix_removed else True
            break  # Only remove the first matching suffix

    # Remove only **one** leading and one trailing underscore, if present
    if prefix_removed or suffix_removed:
        if func_name.startswith("_") and prefix_removed:
            func_name = func_name[1:]
        if func_name.endswith("_") and suffix_removed:
            func_name = func_name[:-1]

    return func_name


class BaseInplaceUpdateTransformer(BaseTransformer):
    """
    A base class to handle inplace update operations and functionalize them.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        # Regex for inplace methods (e.g., fill_(), add_(), etc.)
        self.inplace_method_regex = re.compile(r"\w*[^_]_$")

    def transform(self):
        # Set parent relationships in the tree (so each node knows its parent)
        set_parents(self.root)
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        # Hardcode #1: Check if the function is "ivy_batch_norm"
        if node.name == "ivy_batch_norm":
            # Filter out the "if" node
            node.body = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, gast.If)
                    and ast_to_source_code(stmt.test).strip() == "training"
                )
            ]

            # Modify the return statement
            for stmt in node.body:
                if isinstance(stmt, gast.Return):
                    stmt.value = gast.Tuple(
                        elts=[
                            stmt.value,
                            gast.Name(
                                id="mean",
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            ),
                            gast.Name(
                                id="var",
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            ),
                        ],
                        ctx=gast.Load(),
                    )

        # Hardcode #2: Check if the function is "forward" method of ivy__BatchNorm class
        elif (
            node.name == "forward" and "_BatchNorm" in self.transformer.object_like.name
        ):
            # Find the return statement
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, gast.Return):
                    # Replace the return statement with an assignment and a return
                    node.body[i] = gast.Assign(
                        targets=[
                            gast.Tuple(
                                elts=[
                                    gast.Name(
                                        id="normalized",
                                        ctx=gast.Store(),
                                        annotation=None,
                                        type_comment=None,
                                    ),
                                    gast.Attribute(
                                        value=gast.Name(
                                            id="self",
                                            ctx=gast.Load(),
                                            annotation=None,
                                            type_comment=None,
                                        ),
                                        attr="running_mean",
                                        ctx=gast.Store(),
                                    ),
                                    gast.Attribute(
                                        value=gast.Name(
                                            id="self",
                                            ctx=gast.Load(),
                                            annotation=None,
                                            type_comment=None,
                                        ),
                                        attr="running_var",
                                        ctx=gast.Store(),
                                    ),
                                ],
                                ctx=gast.Store(),
                            )
                        ],
                        value=stmt.value,
                    )
                    node.body.append(
                        gast.Return(
                            value=gast.Name(
                                id="normalized",
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            )
                        )
                    )
                    break

        return node

    def visit_Return(self, node):
        """
        Check if an inplace operation is the return value.
        If the return value contains an inplace operation,
        it should not be converted into an assignment.
        """
        if isinstance(node.value, gast.Call):
            func_name_node = node.value.func
            clean_name = self.get_function_name(func_name_node)
            if re.match(self.inplace_method_regex, clean_name):
                return node

        # Continue visiting other parts of the return statement
        return self.generic_visit(node)

    def visit_Expr(self, node):
        """
        Transform any standalone inplace method calls that are
        not part of an assignment or return.
        Example: tensor.uniform_(-bound, bound) -> tensor = tensor.uniform_(-bound, bound)
        """
        if isinstance(node.value, gast.Call):
            func_name_node = node.value.func
            clean_name = self.get_function_name(func_name_node)
            # If the function is an inplace method, transform the call into an assignment
            if re.match(self.inplace_method_regex, clean_name):
                parent = getattr(node, "parent", None)

                # If it's a call on an attribute (e.g., self.running_mean.zero_())
                if isinstance(node.value.func, gast.Attribute) and not isinstance(
                    parent, gast.Assign
                ):
                    # The object being modified is the attribute's base (e.g., self.running_mean)
                    target = copy.deepcopy(node.value.func.value)
                    target.ctx = gast.Store()

                    # Recreate the function call on the RHS
                    new_call = gast.Call(
                        func=node.value.func,
                        args=node.value.args,
                        keywords=node.value.keywords,
                    )

                    # Replace the expression with an assignment
                    return gast.Assign(targets=[target], value=new_call)

                # If it's a call on a `Name` (e.g., ivy_erf__frnt_)
                if isinstance(node.value.func, gast.Name) and not isinstance(
                    parent, gast.Assign
                ):
                    # The first argument is the target for inplace ops (e.g., ivy_erf__frnt_(x))
                    if node.value.args:
                        first_arg_node = node.value.args[0]
                        if isinstance(first_arg_node, gast.Call):
                            # Special case where the first argument could be another call in a chained manner
                            # i.e. ivy_trunc_normal_(ivy_data_frnt_(module.weight), std=self.config.initializer_range)
                            # in which case we'd need to extract the inner most arg
                            first_arg_node = self.get_target_for_call(first_arg_node)

                        target = copy.deepcopy(first_arg_node)
                        target.ctx = gast.Store()

                        # Recreate the function call on the RHS
                        new_call = gast.Call(
                            func=node.value.func,
                            args=node.value.args,
                            keywords=node.value.keywords,
                        )

                        # Replace the expression with an assignment
                        return gast.Assign(targets=[target], value=new_call)

        # Leave other expressions unchanged
        return node

    def visit_Call(self, node):
        """
        Transform inplace calls inside function bodies, ensuring they are assigned properly.
        """
        func_name_node = node.func
        clean_name = self.get_function_name(func_name_node)
        if re.match(self.inplace_method_regex, clean_name):
            parent = getattr(node, "parent", None)

            # If it's an inplace method acting on an attribute (e.g., self.num_batches_tracked.add_())
            if isinstance(node.func, gast.Attribute) and not isinstance(
                parent, gast.Assign
            ):
                # The target is the object being modified (the base of the attribute, e.g., self.num_batches_tracked)
                target = copy.deepcopy(node.func.value)
                target.ctx = gast.Store()

                new_call = gast.Call(
                    func=node.func, args=node.args, keywords=node.keywords
                )

                return gast.Assign(targets=[target], value=new_call)

        return node  # Leave non-matching calls unchanged

    def get_function_name(self, func_node):
        """
        Get the function name from either gast.Attribute or gast.Name.
        """
        if isinstance(func_node, gast.Attribute):
            return clean_function_name(func_node.attr)
        elif isinstance(func_node, gast.Name):
            return clean_function_name(func_node.id)
        else:
            return ""

    def get_target_for_call(self, node: gast.Call):
        """
        Get the target node from a gast.Call. A target node can be
        the innermost variable for chained gast.Call nodes
        e.g. ivy_trunc_normal_(ivy_data_frnt_(module.weight), std=self.config.initializer_range)
        or the base attr that is being called on e.g.
        self.running_mean.zero_()
        """
        func_node = node.func
        args_node = node.args[0] if node.args else None
        if args_node:
            if isinstance(args_node, gast.Call):
                return self.get_target_for_call(args_node)
            else:
                return args_node
        else:
            return func_node
