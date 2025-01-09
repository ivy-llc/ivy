# global
import gast

# local
from ...transformers.base_transformer import (
    BaseTransformer,
)
from ....utils.ast_utils import get_attribute_full_name


class BaseRenameTransformer(BaseTransformer):
    def __init__(self, node):
        assert isinstance(
            node, gast.AST
        ), "RenameTransformer only accepts gast.AST as input"
        self.root = node
        self.old_name = ""
        self.new_name = ""

    def rename(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        self.visit(self.root)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id == self.old_name:
            new_node = gast.parse(self.new_name).body[0].value
            return new_node
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_full_name = get_attribute_full_name(node)
        if attr_full_name == self.old_name:
            new_node = gast.parse(self.new_name).body[0].value
            return new_node
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.old_name == node.name:
            node.name = self.new_name
        return node
