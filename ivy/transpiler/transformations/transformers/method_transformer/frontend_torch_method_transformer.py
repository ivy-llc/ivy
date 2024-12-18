# global
import builtins
import gast

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
    get_frontend_class,
    is_frontend_api,
    is_ivy_api,
)
from ....utils.ast_utils import (
    ast_to_source_code,
    is_super_call_node,
    MODULE_TO_ALIAS,
    get_function_vars,
)
from ... import transformer_globals as glob


class FrontendTorchMethodToFunctionConverter(BaseMethodToFunctionConverter):
    """
    A class to convert frontend method calls to function calls in a gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
    ):
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.builtin_modules = BUILTIN_LIKELY_MODULE_NAMES + list(
            MODULE_TO_ALIAS.values()
        )
        class_names = ["torch.Tensor", "torch.Size"]
        self.frontend_cls = {cls: get_frontend_class(cls) for cls in class_names}
        variables, non_locals_and_globals = get_function_vars(self.root)
        self.variables = variables.union(non_locals_and_globals)

    def transform(self):
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
        # NOTE: we should also add frontends class for torch.dtype, torch.device classes. For instance,
        # x.dtype.is_floating_point() should ideally become torch.Dtype(x).is_floating_point()
        for cls_name, frontend_cls in self.frontend_cls.items():
            if is_property(
                right,
                frontend_cls,
                to_ignore=("ivy_array", "dtype", "device"),
            ) and left != ["self", "ivy_array"]:
                new_func = gast.parse(f"{cls_name}.{right}").body[0].value
                new_args = [gast.parse(".".join(left)).body[0].value]
                new_keywords = []
                new_call = gast.Call(
                    func=new_func, args=new_args, keywords=new_keywords
                )
                node = new_call

                # add the transformed method to the CONFLICTING_METHODS set
                glob.CONFLICTING_METHODS.add(right)

                return node

        return node

    def visit_Call(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)
        _CHAINED_CALLS = False
        # If the function being called is an Attribute node and its value is a Call node,
        # and the function is not a call to super, transform the call
        if (
            isinstance(node.func, gast.Attribute)
            and isinstance(node.func.value, gast.Call)
            and any(
                is_method_of_class(node.func.attr, frontend_cls)
                for frontend_cls in self.frontend_cls.values()
            )
            and not is_method_of_class(node.func.attr, self.transformer.object_like)
            and not is_super_call_node(node.func)
            and node.func.attr not in dir(builtins)
            and not node.func.attr.startswith(
                "__"
            )  # and node.func.attr not in dunder methods (eg: x.bar().__init__())
            and not node.func.attr
            == "numel"  # special case for numel (ie: torch.Tensor.shape(..).numel --> numel(torch.Tensor.shape(..) X)
        ):
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

            _CHAINED_CALLS = True
            node = new_outer_call

        # Convert the function name to source code and split it
        func_parts = ast_to_source_code(node.func).strip().split(".")
        left, right = func_parts[:-1], func_parts[-1]

        # Check if any part of the function name should not be transformed
        if not self.is_supported_call_node(node.func) or (
            not _CHAINED_CALLS and len(left) == 0
        ):
            return node

        cls_ = [
            cls_name
            for (cls_name, cls) in self.frontend_cls.items()
            if is_method_of_class(right, cls)
        ][0]

        # Transform the method call to a function call
        if left:  # This is a method call
            new_func = gast.parse(f"{cls_}.{right}").body[0].value
            new_args = [gast.parse(".".join(left)).body[0].value] + node.args
        else:  # This is a function call
            new_func = gast.parse(f"{cls_}.{right}").body[0].value
            new_args = node.args
        new_node = gast.Call(func=new_func, args=new_args, keywords=node.keywords)

        # add the transformed method to the CONFLICTING_METHODS set
        glob.CONFLICTING_METHODS.add(right)

        """
        Motivating example for using the @handle_methods decorator.
        
        class DummyCls:
            def __init__(self, a, b):
                self.arr = a
                ...
            
            def flatten(self):
                ...
            
            def dummy_method(self):
                # This is a conflicting method call.
                self.arr.flatten(..)

        In the above example, the method call to flatten() is conflicting.
        `flatten` could refer to `torch.Tensor.flatten` or `DummyCls.flatten`.
        This decision can only be made at runtime once we infer the actual type of `self.arr`.
        Hence, we add the `@handle_methods` decorator to handle this lazy evaluation.

        another example where this is useful is when the method does not
        appear on the DummyCls but rather on the attribute itself e.g.

        class DummyCls:
            def __init__(self, q, b):
                self.q = q
                ...
            
            def dummy_method(self):
                # This is a conflicting method call.
                self.q.conj(..)
        
        in this case we directly apply the handle_methods decorator so it can decide
        at runtime whether to call the frontend method or the `self.q.conj(...)`
        """

        return new_node

    def _is_super_call(self, node):
        """
        Check if a node represents a call to super.
        """
        return (
            isinstance(node, gast.Attribute)
            and isinstance(node.value, gast.Call)
            and isinstance(node.value.func, gast.Name)
            and node.value.func.id == "super"
        )

    def is_supported_call_node(self, node):
        # Convert the node name to source code and split it
        method_name = ast_to_source_code(node).strip()
        node_parts = method_name.split(".")
        *left, right = node_parts
        # filters complex expressions that appear in nested calls
        # 1) torch.stack(..).expand(..) --> value is gast.Call
        # 2) torch.stack(...)[:10].expand(..) --> value is gast.Subscript
        # these expressions represent nested method calls even though
        # they appear as function calls, hence the additional check.
        is_func_obj = (
            isinstance(node, gast.Attribute)
            and not isinstance(node.value, (gast.Call, gast.Subscript))
            and method_name.startswith("torch.")
        )
        is_dunder_obj = right.startswith("__")
        is_super_call = len(left) == 1 and left[0].startswith("super")
        is_method_of_Size = "shape" in ".".join(left) and is_method_of_class(
            right, self.frontend_cls["torch.Size"]
        )
        # this check is to avoid the following special case:
        # x.shape.numel() --> torch.Tensor.numel(x.shape) WRONG ; torch.Tensor.shape(x).numel() RIGHT
        # `numel` is a method of both torch.Size and torch.Tensor.

        # Check if any part of the node name should not be transformed
        if (
            is_func_obj  # dont transform if the given node is a function call not a method
            or is_dunder_obj  # dont transform if the given node is a dunder method (eg: __setattr__, etc.)
            or is_super_call  # dont transform if the given node is a super() call
            or is_method_of_Size  # dont transform if the given node is a torch.Size method call on x.shape (eg: x.shape.numel())
            or any(
                [
                    (
                        any(
                            [
                                prefix == l
                                for prefix in [
                                    "torch_frontend",
                                    "F",
                                    "nn",
                                    "fx",
                                    "super",
                                    "ivy",
                                    "ivy_array",
                                ]
                            ]
                        )
                        and l
                        not in self.variables  # left prefix is from torch(eg: F.pad, torch_frontend.mean etc.) or ivy(ivy.add, ivy.conv2d etc.) or is super()
                    )
                    or l
                    in self.builtin_modules  # left prefix is from builtin modules (eg: math, itertools etc.)
                    or is_builtin_method(
                        l, to_ignore=self.variables
                    )  # left prefix is a builtin method (eg: 'append', 'sum' etc.). to_ignore is a list of variables defined within the function scope.
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
            or not any(
                is_method_of_class(right, frontend_cls)
                for frontend_cls in self.frontend_cls.values()
            )  # right prefix is not a frontend method or attribute (eg: x.foo, y.bar etc.)
            or left == ["self"]
            and is_method_of_class(
                right, self.transformer.object_like
            )  # right prefix is a custom method or attribute (eg: self.<method name>)
        ):
            return False

        return True
