import ast
import os
import sys
from ast import parse
from importlib.util import resolve_name, module_from_spec
from importlib.abc import SourceLoader
from importlib.machinery import FileFinder


class MyLoader(SourceLoader):
    def __init__(self, fullname, path):
        self.fullname = fullname
        self.path = path

    def get_filename(self, fullname):
        return self.path

    def get_data(self, filename):
        with open(filename) as f:
            data = f.read()

        ast_tree = parse(data)
        transformer = ImportTransformer()
        transformer.visit_ImportFrom(ast_tree)
        transformer.visit_Import(ast_tree)
        transformer.impersonate_import(ast_tree)
        ast.fix_missing_locations(ast_tree)
        return ast.unparse(ast_tree)


IMPORT_CACHE = {}


def clear_cache():
    global IMPORT_CACHE
    IMPORT_CACHE = {}


def _ivy_fromimport(name: str, package=None, mod_globals=None, from_list=(), level=0):
    """
    Handles absolute and relative from_import statmement
    :param name:
    :param package:
    :param mod_globals:
    :param from_list:
    :param level:
    :return:
    """
    module_exist = name != ""
    name = "." * level + name
    module = _ivy_import_module(name, package)
    for entry_name, entry_asname in from_list:
        if entry_name == "*":
            if "__all__" in module.__dict__.keys():
                _all = module.__dict__["__all__"]
            else:
                _all = {
                    k: v for (k, v) in module.__dict__.items() if not k.startswith("__")
                }
            for k, v in _all.items():
                mod_globals[k] = v
            continue
        alias = entry_name if entry_asname is None else entry_asname
        # Handles attributes inside module
        try:
            mod_globals[alias] = module.__dict__[entry_name]
            # In the case this is a module from a package
        except KeyError:
            if module_exist:
                in_name = f"{name}.{entry_name}"
            else:
                in_name = name + entry_name
            mod_globals[alias] = _ivy_import_module(in_name, package)
    return module


def _ivy_absolute_import(name: str, asname=None, mod_globals=None):
    """
    Handles absolute import statement
    :param name:
    :return:
    """
    if asname is None:
        _ivy_import_module(name)
        true_name = name.partition(".")[0]
        module = IMPORT_CACHE[true_name]
    else:
        true_name = asname
        module = _ivy_import_module(name)
    mod_globals[true_name] = module


def _ivy_import_module(name, package=None):
    global IMPORT_CACHE
    absolute_name = resolve_name(name, package)
    try:
        return IMPORT_CACHE[absolute_name]
    except KeyError:
        pass

    path = None
    if "." in absolute_name:
        parent_name, _, child_name = absolute_name.rpartition(".")
        parent_module = _ivy_import_module(parent_name)
        path = parent_module.__spec__.submodule_search_locations
    # TODO We can override this to use our meta path without overriding sys.meta
    for finder in sys.meta_path:
        spec = finder.find_spec(absolute_name, path)
        if spec is not None:
            break
    else:
        msg = f"No module named {absolute_name!r}"
        raise ModuleNotFoundError(msg, name=absolute_name)
    module = module_from_spec(spec)
    IMPORT_CACHE[absolute_name] = module
    spec.loader.exec_module(module)
    if path is not None:
        # Set reference to self in parent, if exist
        setattr(parent_module, child_name, module)
    return module


def _retrive_local_modules():
    ret = []
    wd = os.getcwd()
    for entry in os.scandir(wd):
        if entry.is_file():
            if entry.name.endswith(".py"):
                ret.append(entry.name[:-3])
                continue

        if entry.is_dir():
            if "__init__.py" in os.listdir(wd + "/" + entry.name):
                ret.append(entry.name)
    return ret


local_modules = _retrive_local_modules()


def parse_absolute_fromimport(node: ast.ImportFrom):
    # Not to override absolute imports to other packages
    if node.module.partition(".")[0] not in local_modules:
        return node
    to_import = []
    for entry in node.names:
        to_import.append((entry.name, entry.asname))
    # Return a function call
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_ivy_fromimport", ctx=ast.Load()),
            args=[
                ast.Constant(value=node.module),
                ast.Constant(value=None),
                ast.Call(
                    func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]
                ),
                ast.Constant(value=to_import),
            ],
            keywords=[],
        ),
    )


def parse_relative_fromimport(node: ast.ImportFrom):
    if node.module is None:
        name = ""
    else:
        name = node.module
    to_import = []
    for entry in node.names:
        to_import.append((entry.name, entry.asname))
    # Return a function call
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_ivy_fromimport", ctx=ast.Load()),
            args=[
                ast.Constant(value=name),
                ast.Name(id="__package__", ctx=ast.Load()),
                ast.Call(
                    func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]
                ),
                ast.Constant(value=to_import),
                ast.Constant(value=node.level),
            ],
            keywords=[],
        ),
    )


def _create_assign_to_variable(target, value):
    return ast.Assign(
        targets=[ast.Name(id=target, ctx=ast.Store())],
        value=value,
    )


def _create_fromimport_call(name):
    return ast.Call(
        func=ast.Name(id="_ivy_fromimport", ctx=ast.Load()),
        args=[
            ast.Constant(value=name),
        ],
        keywords=[],
    )


def parse_import(node: ast.Import):
    _local_modules = []
    # We don't want to override imports for outside packages
    for entry in node.names.copy():
        if entry.name.partition(".")[0] in local_modules:
            node.names.remove(entry)
            _local_modules.append(entry)
    return_nodes = []
    # Not to include empty import
    if len(node.names) > 0:
        return_nodes.append(node)
    for node in _local_modules:
        return_nodes.append(
            ast.Expr(
                ast.Call(
                    func=ast.Name(id="_ivy_absolute_import", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=node.name),
                        ast.Constant(value=node.asname),
                        ast.Call(
                            func=ast.Name(id="globals", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                )
            )
        )
    return return_nodes, len(_local_modules) > 0


def _create_attrs_from_node(node, attrs=()):
    # Attrs must be in order
    last_node = node
    for attr in attrs:
        last_node = ast.Attribute(
            value=last_node,
            attr=attr,
            ctx=ast.Load(),
        )
    return last_node


class ImportTransformer(ast.NodeTransformer):
    def __init__(self):
        self.insert_index = 0  # TODO hacky solution for __future__
        self.include_ivy_import = False

    def visit_Import(self, node):
        if isinstance(node, ast.Module):
            self.generic_visit(node)
            return node
        if isinstance(node, ast.Import):
            ret, should_impersonate = parse_import(node)
            if should_impersonate and not self.include_ivy_import:
                self.include_ivy_import = True
            return ret

    def visit_ImportFrom(self, node):
        if isinstance(node, ast.Module):
            self.generic_visit(node)
            return node
        if isinstance(node, ast.ImportFrom):
            self.include_ivy_import = True
            if node.level == 0:
                if node.module is not None and node.module == "__future__":
                    self.insert_index = 1
                return parse_absolute_fromimport(node)
            else:
                return parse_relative_fromimport(node)

    def impersonate_import(self, tree: ast.Module):
        if self.include_ivy_import:
            tree.body.insert(
                self.insert_index,
                ast.ImportFrom(
                    module="ivy.backend_compiler",
                    names=[ast.alias(name="_ivy_fromimport")],
                    level=0,
                ),
            )
            tree.body.insert(
                self.insert_index,
                ast.ImportFrom(
                    module="ivy.backend_compiler",
                    names=[ast.alias(name="_ivy_absolute_import")],
                    level=0,
                ),
            )
        return tree


def compile_backend(backend: str):
    loader_details = (MyLoader, [".py"])
    finder = FileFinder.path_hook(loader_details)
    sys.path_hooks.insert(0, finder)
    sys.path_importer_cache.clear()
    ivy_pack = _ivy_import_module("ivy")
    ivy_pack.set_backend(backend)
    sys.path_hooks.remove(finder)
    return ivy_pack
