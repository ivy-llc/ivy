import astunparse
import ast

import sys
import subprocess
import os

_reference_backend = ""
_target_backend = ""


_backend_ref_name = "tf"
_target_backend_name = "torch"

# This should be imported in the module
_not_imlpemented_exc_name = "NotImplementedError"
_decorator_black_list = [
    "with_unsupported_dtypes",
    "with_supported_dtypes",
    "with_unsupported_devices",
    "with_supported_devices",
    "with_unsupported_device_and_dtypes",
    "with_supported_device_and_dtypes",
]

type_mapping = {}


class SourceTransformer(ast.NodeTransformer):
    def __init__(self, type_map):
        self.type_map = type_map
        self.registered_imports = set()

    def _get_full_name(self, node):
        return astunparse.unparse(node)

    def visit_Name(self, node: ast.Name):
        try:
            node.id = self.type_map[node.id]
        except KeyError:
            pass
        else:
            self.registered_imports.add(node.id)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        try:
            str_repr = self._get_full_name(node).strip()
            new_node = ast.parse(self.type_map[str_repr])
            node = new_node.body[0].value
        except KeyError:
            # Do not remove original framework type hints
            pass
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign):
        for name in node.targets:
            if name.id.startswith("_") and not name.id.endswith("__"):
                return None
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Remove private functions
        if node.name.startswith("_") and not node.name.endswith("__"):
            self.generic_visit(node)
            return None
        else:
            # Replace function body with Pass
            node.body = [
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id=_not_imlpemented_exc_name, ctx=ast.Load()),
                        args=[
                            ast.Constant(value="Function Not Implemented", kind=None)
                        ],
                        keywords=[],
                    ),
                    cause=None,
                )
            ]
            # Update decorators not to include ones in the blacklist
            # Add Not Implemented decorator
            new_list = []
            for entry in node.decorator_list:
                if isinstance(entry, ast.Call):
                    name_of_decorator = entry.func.id
                else:
                    name_of_decorator = entry.id
                if name_of_decorator in _decorator_black_list:
                    continue
                new_list.append(entry)
            node.decorator_list = new_list
        self.generic_visit(node)
        return node


# Modify the AST tree
def _parse_module(tree: ast.Module) -> ast.Module:
    transformer = SourceTransformer(type_mapping)
    transformer.visit(tree)

    import_node = ast.Import(names=[ast.alias(name=_target_backend_name, asname=None)])
    tree.body.insert(0, import_node)

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

            ref_tree = ast.parse(ref_str)
            try:
                tree_to_write = _parse_module(ref_tree)
            except Exception:
                print(f"Failed to parse {os.path.join(root, name)}")

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

    subprocess.run(["black", backend_generation_path])
    subprocess.run(
        [
            "autoflake",
            "-i",
            "--remove-all-unused-imports",
            "--quiet",
            "-r",
            backend_generation_path,
        ]
    )


# TODO remove
if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2])
