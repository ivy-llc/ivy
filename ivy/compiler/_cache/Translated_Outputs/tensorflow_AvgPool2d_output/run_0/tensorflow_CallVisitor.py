import ast

from .tensorflow__helpers import tensorflow_store_config_info


class tensorflow_CallVisitor(ast.NodeVisitor):
    @tensorflow_store_config_info
    def __init__(self):
        self.func_name = None

    def visit_Call(self, node):
        self.func_name = ast.unparse(node.func).strip()
        return super().generic_visit(node)
