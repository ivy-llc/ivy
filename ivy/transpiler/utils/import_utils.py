# global
import gast
import importlib.util
import inspect
from pathlib import Path
import sys
import types
from typing import Callable, Iterable, Set, List, Any, Union

# local
from .ast_utils import ast_to_source_code
from ..transformations.transformers.base_transformer import (
    BaseTransformer,
)


def load_module_from_path(file_path):
    # Ensure file path is resolved and convert to Path object
    file_path = Path(file_path).resolve()

    # Check if the file already has a .py extension, and if not, add it
    if not file_path.suffix == ".py":
        file_path = file_path.with_suffix(".py")

    # Extract the module name from the file path (stem is filename without extension)
    module_name = file_path.stem

    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Insert the module into sys.modules and execute it
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def combine_imports(old_imports, new_imports):
    new_imports_list = new_imports.split("\n")

    for import_stmt in new_imports_list:
        if import_stmt not in old_imports:
            old_imports += "\n" + import_stmt

    return old_imports


def split_imports_and_code(source):
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "import" not in line:
            break
    imports = "\n".join(lines[:i])
    code = "\n".join(lines[i:]) + "\n"
    return imports, code


def inject_standard_imports():
    # TODO: Make this function generic. The transformers below
    # can be used
    import_statements = [
        "import ivy.functional.frontends.torch as torch",
        "import ivy.functional.frontends.torch.nn as nn",
        "import ivy.functional.frontends.torch.nn.functional as F",
        "from ivy.functional.frontends.torch import Tensor",
        "import ivy",
    ]
    return "\n".join(import_statements) + "\n\n"


def inject_module_dependencies(translated_strings, cacher, callable_obj):
    # TODO: Make this function generic. The transformers below
    # can be used.
    helper_imports = []
    other_imports = []
    for translated_name in translated_strings:
        orig_name = translated_name.replace("Translated_", "")
        for func_weakref in cacher._all_translated_objs.data.keys():
            func = func_weakref()
            if func and func.__qualname__.split(".")[-1] == orig_name:
                if inspect.isfunction(func):
                    if not inspect.isfunction(callable_obj):
                        helper_imports.append(f"from helpers import {translated_name}")
                else:
                    other_imports.append(f"from {orig_name} import {translated_name}")
                break

    helper_imports.sort()
    other_imports.sort()

    imports = other_imports + helper_imports

    return "" if not imports else "\n".join(imports) + "\n\n"


def inject_builtin_imports(translator):
    import_statements = []
    for mod in translator._imports:
        import_statements.append(f"import {mod}")

    return "" if not import_statements else "\n".join(import_statements) + "\n\n"


class ModuleImportsFetcher(BaseTransformer):
    """
    Given a set of relative and absolute imports as strings, this transformer
    is responsible for fetching all the import nodes from a given module that
    match the representations of the passed in import strings. This is used
    for example when you want to fetch the exact matching import nodes from
    a given module. Mostly used internally by the ImportsRetriever.
    """

    def visit_Module(self, node):
        import_nodes = []
        for child in node.body:
            if isinstance(child, (gast.Import, gast.ImportFrom)):
                import_nodes.append(child)
                continue

        for import_node in import_nodes:
            add_import = False

            # Relative imports
            if isinstance(import_node, gast.ImportFrom):
                new_names = []
                for alias in import_node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name in self._relative_imports:
                        new_names.append(alias)
                        add_import = True

                # For relative imports, fix the leading . imports
                module = import_node.module or ""
                level = import_node.level
                if level != 0 and add_import:
                    import_node.module = ".".join(
                        self._module.__name__.rsplit(".", level)[:1]
                    )
                    if module:
                        import_node.module += "." + module
                    import_node.level = 0

            # Absolute imports
            if isinstance(import_node, gast.Import):
                new_names = []
                for alias in import_node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name in self._absolute_imports:
                        new_names.append(alias)
                        add_import = True

            # Append the new import node
            if add_import:
                import_node.names = new_names
                self.import_nodes.append(import_node)

        return node

    def fetch_matching_imports(
        self,
        module: types.ModuleType,
        relative_imports: Iterable[str],
        absolute_imports: Iterable[str],
    ) -> List[Union[gast.Import, gast.ImportFrom]]:
        """Interface to be used to fetch all the import nodes
        as is from a given module, provided an iterable of
        absolute and relative import representations in
        string form."""
        self._module = module
        self._relative_imports = relative_imports
        self._absolute_imports = absolute_imports
        self.import_nodes = list()
        module_source = inspect.getsource(self._module)
        tree = gast.parse(module_source)
        tree = self.visit_Module(tree)
        return self.import_nodes


class ImportsRetriever(BaseTransformer):
    """
    Transformer to generate a set of import nodes for the given `obj`
    i.e. it looks in the module of `obj` and retrieves all the import
    nodes that are being used by `obj`.
    """

    def __init__(self, obj: Callable, module=None) -> None:
        self._obj = obj
        self._module = inspect.getmodule(obj) if not module else module
        self.relative_imports: Set[str] = set()
        self.absolute_imports: Set[str] = set()
        self.import_nodes: List[Any] = list()

    def _add_absolute_import(self, module):
        self.absolute_imports.add(module)

    def _add_relative_import(self, attr):
        if attr in self.absolute_imports:
            return
        self.relative_imports.add(attr)

    def _set_top_level_namespace_attr(self, node, name):
        if isinstance(node, gast.Attribute):
            if isinstance(node.value, gast.Name):
                node.value.id = name
        elif isinstance(node, gast.Name):
            node.id = name
        return node

    def _get_top_level_namespace_attr(self, node):
        if isinstance(node, gast.Attribute):
            if isinstance(node.value, gast.Name):
                return node.value.id
        elif isinstance(node, gast.Name):
            return node.id
        else:
            return node.name

    def _store_module_for_node(self, node):
        if isinstance(node, gast.Attribute):
            attr = self._get_top_level_namespace_attr(node)
            if attr not in self._module.__dict__:
                return node
            maybe_module = self._module.__dict__[attr]
            if inspect.ismodule(maybe_module):
                if hasattr(maybe_module, node.attr):
                    self._add_absolute_import(maybe_module.__name__)
                else:
                    self._add_relative_import(node.attr)
                return node
            self._add_relative_import(node.attr)
        elif isinstance(node, gast.Name):
            attr = self._get_top_level_namespace_attr(node)
            if attr not in self._module.__dict__:
                return node
            self._add_relative_import(attr)
        else:
            unparsed = ast_to_source_code(node)
            split = unparsed.rsplit(".", 1)
            if split[0] not in self._module.__dict__:
                if isinstance(node.func, gast.Name):
                    attr = node.func.id
                    if attr not in self._module.__dict__:
                        return node
                    self._add_relative_import(attr)
                return node

            if len(split) != 2:
                self._add_relative_import(split[0])
            else:
                module_name, attr = split
                module = self._module.__dict__[module_name]
                if "." in module.__name__:
                    module, attr = module.__name__.rsplit(".", 1)
                    self._add_relative_import(attr)
                    return node
                if inspect.ismodule(module):
                    self._add_absolute_import(module.__name__)
        return node

    def visit_Attribute(self, node):
        node = self._store_module_for_node(node)
        node = self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node = self._store_module_for_node(node)
        node = self.generic_visit(node)
        return node

    def visit_Call(self, node):
        node = self._store_module_for_node(node)
        node = self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node) -> Any:
        if node.name == self._obj.__name__:
            node = self.generic_visit(node)
            imports_fetcher = ModuleImportsFetcher()
            self.import_nodes = imports_fetcher.fetch_matching_imports(
                self._module, self.relative_imports, self.absolute_imports
            )
            return node
        return node

    def retrieve_imports(self, node) -> List[Union[gast.Import, gast.ImportFrom]]:
        """Interface to be used to retrieve all the import nodes
        as is from the module of a given callable `obj` that are
        referenced to in the definition of `obj`."""
        self.visit(node)
        return self.import_nodes
