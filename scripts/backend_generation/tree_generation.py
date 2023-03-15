import astunparse
import ast

# import sys
import os

_reference_backend = ""
_target_backend = ""


_backend_ref_name = "tf"
_target_backend_name = "torch"

# This should be imported in the module
_not_implemented_decorator_name = "not_implemented"
_decorator_black_list = [
    "with_unsupported_dtypes",
    "with_supported_dtypes",
    "with_unsupported_devices",
    "with_supported_devices",
    "with_unsupported_device_and_dtypes",
    "with_supported_device_and_dtypes",
]

type_mapping = {
    "int": "float",
    "Tensor": "Array",
    "Variable": "VarArray",
    "Scaler": "NotAscaler",
}


class ImportTransformer(ast.NodeTransformer):
    # TODO add backend specific imports for the type hints
    def visit_Import(self, node: ast.Import):
        if isinstance(node, ast.Import):
            # Only keep "import typing" statements
            entries_list = []
            for entry in node.names:
                if entry.name.startswith("typing"):
                    entries_list.append(entry)
            if len(entries_list) > 0:
                node.names = entries_list
            else:
                self.generic_visit(node)
                return None
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Only keep imports from "typing"
        if isinstance(node, ast.ImportFrom):
            if not node.module == "typing":
                self.generic_visit(node)
                return None
        self.generic_visit(node)
        return node


class FunctionBodyTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if isinstance(node, ast.FunctionDef):
            # Remove private functions
            if node.name.startswith("_") and not node.name.endswith("__"):
                self.generic_visit(node)
                return None
            else:
                # Replace function body with Pass
                node.body = [ast.Pass()]
                # Update decorators not to include ones in the blacklist
                # Add Not Implemented decorator
                new_list = [ast.Name(_not_implemented_decorator_name, ctx=ast.Load())]
                for entry in node.decorator_list:
                    if entry.id in _decorator_black_list:
                        continue
                    new_list.append(entry)
                node.decorator_list = new_list
        self.generic_visit(node)
        return node


class TypeHintTransformer(ast.NodeTransformer):
    def __init__(self, type_map):
        self.type_map = type_map
        self.registered_type_hints = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node, ast.Name):
            if node.id == _backend_ref_name:
                node.id = _target_backend_name
            else:
                try:
                    node.id = self.type_map[node.id]
                except KeyError:
                    pass
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node, ast.Attribute):
            try:
                node.attr = self.type_map[node.attr]
            except KeyError:
                pass
            else:
                # Function Scope-Level
                self.registered_type_hints.add(node.attr)
        self.generic_visit(node)
        return node


# Modify the AST tree
def _parse_module(tree: ast.Module) -> ast.Module:
    # Walk the AST tree and update type hints
    transformer = TypeHintTransformer(type_mapping)
    transformer.visit_Name(tree)
    print("Found:", transformer.registered_type_hints)

    # Update decorators, update function body, remove private functions
    FunctionBodyTransformer().visit_FunctionDef(tree)
    # Update imports, remove backend specific imports
    ImportTransformer().visit_Import(tree)
    # Add target backend import, add type hints classes imports
    ast.fix_missing_locations(tree)
    return tree


def _copy_tree(backend_reference_path: str, backend_generation_path: str):
    for root, _, files in os.walk(backend_reference_path):
        # Skip pycache dirs
        if root.endswith("__pycache__"):
            continue

        relative_path = os.path.relpath(root, backend_reference_path)
        # Make backend dirs
        os.makedirs(os.path.join(backend_generation_path, relative_path))

        for name in files:
            # Skip pycache modules
            if name.endswith("pyc"):
                continue

            with open(os.path.join(root, name)) as ref_file:
                # Read source file from reference backend
                ref_str = ref_file.read()

            tree_to_write = _parse_module(ast.parse(ref_str))

            # Create target backend
            with open(
                os.path.join(backend_generation_path, relative_path, name), "w"
            ) as generated_file:
                generated_file.write(astunparse.unparse(tree_to_write))


def generate(backend_reference: str, target_backend: str):
    # Set backend variables
    global _target_backend, _reference_backend
    _target_backend = target_backend
    _reference_backend = backend_reference

    backends_root = "../../ivy/functional/backends/"
    backend_reference_path = backends_root + backend_reference
    backend_generation_path = backends_root + target_backend

    # Copy and generate backend tree
    _copy_tree(backend_reference_path, backend_generation_path)


# TODO remove
if __name__ == "__main__":

    with open("func2.py") as ref_file:
        # Read source file from reference backend
        ref_str = ref_file.read()

    tree = ast.parse(ref_str)
    tree_to_write = _parse_module(tree)
    print(astunparse.unparse(tree))
    # generate(sys.argv[1], sys.argv[2])
