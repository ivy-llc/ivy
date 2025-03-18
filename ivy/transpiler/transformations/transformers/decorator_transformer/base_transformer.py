# globals
import types

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
import gast
from ...transformer import Transformer
from ...transformers.base_transformer import (
    BaseTransformer,
)
from ....utils.ast_utils import ast_to_source_code
from ....utils.type_utils import Types


class BaseDecoratorRemover(BaseTransformer):
    """
    A class to remove all decorators from a given function.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.disallowed_decorators = set(
            [
                "_copy_to_script_wrapper",
                "torch.jit.ignore",
                "torch._jit_internal._overload_method",
            ]
        )

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        new_decorator_list = []
        if self.transformer.object_like.type == Types.FunctionType:
            # remove the property decorator if we are translating an ivy.Array/frontend Tensor property
            self.disallowed_decorators = self.disallowed_decorators.union(
                set(["property", "staticmethod"])
            )
        for decorator in node.decorator_list:
            if isinstance(decorator, (gast.Name, gast.Attribute)):
                if (
                    ast_to_source_code(decorator).strip()
                    not in self.disallowed_decorators
                ):
                    new_decorator_list.append(decorator)
            elif isinstance(decorator, gast.Call):
                if (
                    ast_to_source_code(decorator.func).strip()
                    not in self.disallowed_decorators
                ):
                    new_decorator_list.append(decorator)
            elif isinstance(decorator, gast.Attribute):
                if (
                    ast_to_source_code(decorator).strip()
                    not in self.disallowed_decorators
                ):
                    new_decorator_list.append(decorator)
            else:
                new_decorator_list.append(decorator)
        node.decorator_list = new_decorator_list
        self.generic_visit(node)
        return node
