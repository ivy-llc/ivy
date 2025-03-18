# local
import gast
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import (
    ast_to_source_code,
)
from ...transformers.postprocessing_transformer.base_transformer import (
    BaseCodePostProcessor,
)


class NativeTorchCodePostProcessor(BaseCodePostProcessor):
    """
    A class to perform post-processing the final gast AST.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        self.visit(self.root)

    def visit_ClassDef(self, node):
        self.generic_visit(node)

        if not node.body:
            node.body = [gast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        if not node.body:
            node.body = [gast.Pass()]
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_str = ast_to_source_code(node).strip()
        if attr_str in self.configuration.tensor_cls_map:
            return gast.parse(self.configuration.tensor_cls_map[attr_str]).body[0].value
        elif attr_str in self.configuration.torch_meta:
            return gast.Constant(
                value=self.configuration.torch_meta[attr_str], kind=None
            )
        return node
