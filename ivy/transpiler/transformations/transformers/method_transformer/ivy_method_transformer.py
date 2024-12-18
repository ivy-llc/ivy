# global
import gast
import ivy
import re

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.conversion_utils import (
    BUILTIN_LIKELY_MODULE_NAMES,
)
from ..method_transformer.base_transformer import (
    BaseMethodToFunctionConverter,
)
from ....utils.api_utils import (
    is_property,
    is_builtin_method,
    is_method_of_class,
    is_frontend_api,
    is_ivy_api,
    get_function_from_modules,
)
from ....utils.ast_utils import (
    ast_to_source_code,
    is_super_call_node,
    MODULE_TO_ALIAS,
    get_function_vars,
)
from ... import transformer_globals as glob


class IvyMethodToFunctionConverter(BaseMethodToFunctionConverter):
    """
    A class to convert ivy method calls to function calls in a gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        class_name="ivy.Array",
    ):
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.class_name = class_name
        self.builtin_modules = BUILTIN_LIKELY_MODULE_NAMES + list(
            MODULE_TO_ALIAS.values()
        )
        self.ivy_cls = ivy.Array
        variables, non_locals_and_globals = get_function_vars(self.root)
        self.variables = variables.union(non_locals_and_globals)
        if self.transformer.target == "tensorflow":
            self.properties_to_ignore = ("data", "shape", "dtype", "device", "strides")
        elif self.transformer.target in ("jax", "numpy"):
            # jax exposes all these attributes on the ArrayImpl class.
            self.properties_to_ignore = (
                "data",
                "shape",
                "dtype",
                "device",
                "strides",
                "size",
                "itemsize",
                "ndim",
            )

    def transform(self):
        # no need to transform method calls for backend impl in ivy.
        # eg: x.T would mean tf.Tensor.T OR jax.Array.T in the backend.
        # Hence, we should not transform it to be ivy.Array.T
        if not self.transformer.object_like.is_backend_api:
            self.visit(self.root)

    def visit_Assign(self, node):
        # Apply the visit_Attribute logic to the RHS of the assignment
        node.value = self.visit(node.value)
        return node

    def visit_Attribute(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)

        # Convert the attribute name to source code and split it
        attr_parts = ast_to_source_code(node).strip().split(".")
        left, right = attr_parts[:-1], attr_parts[-1]

        # Check if any part of the attribute name should not be transformed
        if not self.is_supported_call_node(node):
            return node

        # Transform the attribute to a function call
        # NOTE: maybe we should also transform `shape` property for ivy arrays (eg: x.shape --> ivy.Array.shape(x))
        # this can come in handy to couple dynamic/static shape handling in a single place
        if is_property(
            right,
            self.ivy_cls,
            to_ignore=self.properties_to_ignore,
        ) and left != ["self", "_data"]:
            new_func = gast.parse(f"{self.class_name}.{right}").body[0].value
            new_args = [gast.parse(".".join(left)).body[0].value]
            new_keywords = []
            new_call = gast.Call(func=new_func, args=new_args, keywords=new_keywords)
            node = new_call

            # add the transformed method to the CONFLICTING_METHODS set
            glob.CONFLICTING_METHODS.add(right)

        return node

    def visit_Call(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)
        _CHAINED_CALLS = False
        call_name = ast_to_source_code(node).strip()
        is_jax_at_call = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*\.at\[[^\]]+\]\.(min|max|sum|prod)", call_name
        )
        # If the function being called is an Attribute node and its value is a Call node,
        # and the function is not a call to super, transform the call
        if (
            isinstance(node.func, gast.Attribute)
            and isinstance(node.func.value, gast.Call)
            and is_method_of_class(node.func.attr, self.ivy_cls)
            and not is_method_of_class(node.func.attr, self.transformer.object_like)
            and not is_super_call_node(node.func)
            and not is_jax_at_call
            and not node.func.attr.startswith(
                "__"
            )  # and node.func.attr not in dunder methods (eg: x.bar().__init__())
        ):
            if any(
                substr in ast_to_source_code(node.func)
                for substr in ("current_backend", "to_numpy")
            ):
                return node
            # Get the inner and outer calls
            inner_call = node.func.value
            outer_call = node

            # Create a new Call node for the outer call, but with the inner call as its function
            new_outer_call = gast.Call(
                func=gast.Name(
                    id=node.func.attr,
                    ctx=gast.Load(),
                    type_comment=None,
                    annotation=None,
                ),
                args=[inner_call] + outer_call.args,
                keywords=outer_call.keywords,
            )

            node = new_outer_call
            _CHAINED_CALLS = True

        # Convert the function name to source code and split it
        func_parts = ast_to_source_code(node.func).strip().split(".")
        left, right = func_parts[:-1], func_parts[-1]

        # Check if any part of the function name should not be transformed
        if not self.is_supported_call_node(node.func) or (
            not _CHAINED_CALLS and len(left) == 0
        ):
            return node

        # Transform the method call to a function call
        if left:  # This is a method call
            new_func = gast.parse(f"{self.class_name}.{right}").body[0].value
            new_args = [gast.parse(".".join(left)).body[0].value] + node.args
        else:  # This is a function call
            new_func = gast.parse(f"{self.class_name}.{right}").body[0].value
            new_args = node.args
        new_node = gast.Call(func=new_func, args=new_args, keywords=node.keywords)

        # add the transformed method to the CONFLICTING_METHODS set
        glob.CONFLICTING_METHODS.add(right)

        return new_node

    def _is_super_call(self, node):
        """
        Check if a node represents a call to super.
        """
        return (
            isinstance(node, gast.Attribute)
            and isinstance(node.value, gast.Call)
            and isinstance(node.value.func, gast.Name)
            and "super" in node.value.func.id
        )

    def is_supported_call_node(self, node):
        # Convert the node name to source code and split it
        method_name = ast_to_source_code(node).strip()
        node_parts = method_name.split(".")
        *left, right = node_parts
        # filters complex expressions that appear in nested calls
        # 1) ivy.stack(..).expand(..) --> value is gast.Call
        # 2) ivy.stack(...)[:10].expand(..) --> value is gast.Subscript
        # these expressions represent nested method calls even though
        # they appear as function calls, hence the additional check.
        is_func_obj = (
            isinstance(node, gast.Attribute)
            and not isinstance(node.value, (gast.Call, gast.Subscript))
            and method_name.startswith("ivy.")
        )
        is_curr_bknd_obj = (
            isinstance(node, gast.Attribute)
            and isinstance(node.value, gast.Call)
            and isinstance(node.value.func, gast.Name)
            and node.value.func.id == "current_backend"
        )
        is_dunder_obj = right.startswith("__")
        is_super_call = len(left) == 1 and left[0].startswith("super")
        is_jax_at_call = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*\.at\[[^\]]+\]\.(min|max|sum|prod)", method_name
        )
        # Check if any part of the node name should not be transformed
        if (
            is_func_obj  # dont transform if the given node is a function call not a method
            or is_curr_bknd_obj  # dont transform if the given node is a current_backend(x).<some_func> call
            or is_dunder_obj  # dont transform if the given node is a dunder method (eg: __setattr__, etc.)
            or is_super_call  # dont transform if the given node is a super() call
            or is_jax_at_call  # dont transform if the given node is a jax.numpy.at[0].(min|max|sum|prod) call
            or any(
                [
                    (
                        any(
                            [
                                prefix == l
                                for prefix in [
                                    "super",
                                    "tf",
                                    "tensorflow",
                                    "jax",
                                    "jnp",
                                    "jax.numpy",
                                    "jaxlib",
                                    "current_backend",
                                ]
                            ]
                        )
                        and l
                        not in self.variables  # left prefix is from ivy backend(eg: tf.pad(...), jnp.dtype(...) etc.) or ivy.current_backend(ivy.current_backend().add, etc.) or is super()
                    )
                    or l
                    in self.builtin_modules  # left prefix is from builtin modules (eg: math, itertools etc.)
                    or is_builtin_method(
                        l, to_ignore=self.variables
                    )  # left prefix is a builtin method (eg: math.prod, itertools.chain etc.)
                    or left == ["self"]
                    and not any(
                        (
                            self.transformer.object_like.is_frontend_api,
                            self.transformer.object_like.is_ivy_api,
                        )
                    )  # left prefix is a method from a user class (eg: self.<some method>() inside a func/class thats not from ivy/frontend )
                    for l in left
                ]
            )
            or right == "current_backend"  # appears as a method on ivy.Array as well
            or not is_method_of_class(
                right,
                self.ivy_cls,
                to_ignore=(
                    "ivy.data_classes.array.wrapping",
                ),  # NOTE: this is a bug. certain methods are missing on ivy.Array(eg:ivy.Array.ones) and hence point here(https://github.com/ivy-llc/ivy/blob/28ed6a7544eb4a68583fb6d6785143e66af62e08/ivy/data_classes/array/wrapping.py#L61)
            )  # is not an ivy method or attribute
            or left == ["self"]
            and is_method_of_class(
                right, self.transformer.object_like
            )  # is a custom method or attribute
        ):
            return False

        return True
