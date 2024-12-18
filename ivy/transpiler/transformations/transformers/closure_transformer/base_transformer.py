# local
import gast
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ...transformers.base_transformer import (
    BaseTransformer,
)


class BaseClosureToLocalTransformer(BaseTransformer):
    """
    A class to convert variables in a function into local variables.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        self._inject_closure_vars(node)
        self.generic_visit(node)
        return node

    def _inject_closure_vars(self, node):
        closure_vars = self.transformer.object_like.closure_vars
        for var_name, var_value in closure_vars.items():
            if callable(var_value):
                continue
            # Create an assignment node for each closure variable
            assign_node = gast.Assign(
                targets=[
                    gast.Name(
                        id=var_name,
                        ctx=gast.Store(),
                        type_comment=None,
                        annotation=None,
                    )
                ],
                value=gast.Constant(value=var_value, kind=None),
            )
            # Insert the assignment node at the start of the function's body
            node.body.insert(0, assign_node)
