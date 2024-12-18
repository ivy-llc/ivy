import gast

# local
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.base_transformer import (
    BaseTransformer,
)


class BaseProfilingTransformer(BaseTransformer):
    """
    A class to add profiling to the given gast AST.
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

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        decorator_list = node.decorator_list
        decorator_list.append(
            gast.Name(
                id="profiling.profiling_logging_decorator",
                ctx=gast.Load(),
                annotation=None,
                type_comment=None,
            )
        )
        node.decorator_list = decorator_list
        return node
