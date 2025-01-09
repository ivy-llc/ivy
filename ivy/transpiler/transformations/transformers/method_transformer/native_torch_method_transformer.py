# global
import gast
import importlib

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
    is_method_of_class,
    get_hf_class,
)
from ....utils.ast_utils import (
    ast_to_source_code,
    MODULE_TO_ALIAS,
)


# NOTE: this approach is based on a temporary solution where we only translate methods of
# utility classes eg: ModuleUtilsMixin, GenerationMixin on a need-per basis. Ideally,
# if/when we are able to successfully translate the entire class and **all** of its
# associated methods, this transformer will become obsolete and can therefore be removed.
class NativeTorchMethodToFunctionConverter(BaseMethodToFunctionConverter):
    """
    A class to convert utility class method calls to function calls in a gast AST.
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
        class_names = ["ModuleUtilsMixin", "GenerationMixin"]
        self.hf_class = {cls: get_hf_class(cls) for cls in class_names}

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        # Recursively visit child nodes
        self.generic_visit(node)

        # Convert the function name to source code and split it
        func_parts = ast_to_source_code(node.func).strip().split(".")
        left, right = func_parts[:-1], func_parts[-1]

        # Check if any part of the function name should not be transformed
        if not self.is_supported_call_node(node.func):
            return node

        cls_ = [
            cls_name
            for (cls_name, cls) in self.hf_class.items()
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

        self.transformer.object_module = importlib.import_module(
            "transformers.modeling_utils"
        )
        return new_node

    def is_supported_call_node(self, node):
        # Convert the node name to source code and split it
        method_name = ast_to_source_code(node).strip()
        node_parts = method_name.split(".")
        *left, right = node_parts
        if left == ["self"] and any(
            is_method_of_class(right, hf_class) for hf_class in self.hf_class.values()
        ):
            return True
        return False
