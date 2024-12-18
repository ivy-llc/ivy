# global
import gast
import os

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import (
    ast_to_source_code,
)
from ....utils.api_utils import is_ivy_api
from .base_transformer import (
    BaseCodePostProcessor,
)


class FrontendTorchCodePostProcessor(BaseCodePostProcessor):
    """
    A class to perform post-processing the final gast AST.
    This involves replacing all `torch.Tensor` names with `ivy.Array`.
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

    def _maybe_replace_torch_tensor_type_check(self, arg):
        """
        Transform the type check argument of an isinstance call
        to replace Tensor with (ivy.Array, ivy.Array).
        """

        def _replace_tensor_with_ivy_array(elt):
            if ast_to_source_code(elt).strip() in {"Tensor", "torch.Tensor"}:
                return [
                    gast.Attribute(
                        value=gast.Name(id="ivy", ctx=gast.Load()),
                        attr="Array",
                        ctx=gast.Load(),
                    ),
                    gast.Attribute(
                        value=gast.Name(id="ivy", ctx=gast.Load()),
                        attr="Variable",
                        ctx=gast.Load(),
                    ),
                ]
            return [elt]

        # If the argument is a tuple, transform each element
        if isinstance(arg, gast.Tuple):
            transformed_elts = []
            for elt in arg.elts:
                transformed_elts.extend(_replace_tensor_with_ivy_array(elt))
            return gast.Tuple(elts=transformed_elts, ctx=gast.Load())
        else:
            # Handle the single type case
            transformed_elts = _replace_tensor_with_ivy_array(arg)
            return gast.Tuple(elts=transformed_elts, ctx=gast.Load())

    def transform(self):
        self.visit(self.root)

    def visit_Name(self, node):
        self.generic_visit(node)
        if self.transformer.object_like.is_ivy_api:
            if node.id == "self":
                node.id = self.new_name

        attr_str = ast_to_source_code(node).strip()
        if attr_str == 'Tensor':
            return gast.parse("ivy.Array").body[0].value
        elif attr_str == 'Parameter':
            return gast.parse("ivy.Variable").body[0].value
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_str = ast_to_source_code(node).strip()
        if attr_str in self.configuration.array_and_module_map:
            return (
                gast.parse(self.configuration.array_and_module_map[attr_str])
                .body[0]
                .value
            )
        elif attr_str in self.configuration.dtype_mapping:
            return gast.parse(self.configuration.dtype_mapping[attr_str]).body[0].value
        if self.transformer.object_like.is_ivy_api:
            if isinstance(node.value, gast.Name) and node.value.id == "self":
                node.value.id = self.new_name
            if node.attr in ("data", "_data", "ivy_array", "_ivy_array"):
                return node.value
        return node

    def visit_Call(self, node):
        func_name = ast_to_source_code(node.func).strip()
        if func_name == "isinstance":
            # Check if the call is to isinstance and if so,
            # maybe transform the type check argument to
            # convert `isinstance(..., (torch.Tensor))` to
            # `isinstance(..., (ivy.Array, ivy.Array))` calls
            # which can later be lowered to
            # `isinstance(..., (tensorflow.Tensor, tensorflow.Variable))`
            node.args[1] = self._maybe_replace_torch_tensor_type_check(node.args[1])

        self.generic_visit(node)

        # if func_name == "ivy.inplace_update":
        #     node = gast.Attribute(value=node, attr="data", ctx=gast.Load)
        func_name = ast_to_source_code(node.func).strip()
        if func_name == 'ivy.Variable':
            # 1) Filter out 'requires_grad' from keyword arguments
            node.keywords = [
                kwarg for kwarg in node.keywords if kwarg.arg != "requires_grad"
            ]
            # 2) Convert 'data' keyword argument (if it exists) to a positional argument 
            for kwarg in node.keywords:
                if kwarg.arg == "data":
                    node.args.append(kwarg.value)
            node.keywords = [
                kwarg for kwarg in node.keywords if kwarg.arg != "data"
            ]

        elif (
            ast_to_source_code(node).strip() == "np.log2(8)"
        ):  # TODO: remove this hardcode and properly handle dunder methods
            return gast.parse("np.log2(8).astype('float32')").body[0].value
        elif (
            ast_to_source_code(node).strip() == "np.zeros((channels, channels))"
        ):  # TODO: remove this hardcode and properly handle dunder methods
            return (
                gast.parse("np.zeros((channels, channels),dtype='float32')")
                .body[0]
                .value
            )
        return node
