import ast


class tensorflow_CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func_name = None

    def visit_Call(self, node):
        self.func_name = ast.unparse(node.func).strip()
        return super().generic_visit(node)
