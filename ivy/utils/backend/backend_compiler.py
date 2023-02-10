import ast
import os
import sys
import traceback
import inspect
from ast import parse
from importlib.util import resolve_name, module_from_spec, spec_from_file_location
from importlib.abc import Loader, MetaPathFinder

IMPORT_CACHE = {}

# AST helpers ##################


def _parse_absolute_fromimport(node: ast.ImportFrom):
    # Not to override absolute imports to other packages
    if node.module.partition(".")[0] not in LOCAL_MODULES:
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
                _create_list(to_import),
            ],
            keywords=[],
        ),
    )


def _parse_relative_fromimport(node: ast.ImportFrom):
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
                _create_list(to_import),
                ast.Constant(value=node.level),
            ],
            keywords=[],
        ),
    )


def _create_list(elements):
    _elts = [ast.Constant(value=element) for element in elements]
    return ast.List(elts=_elts, ctx=ast.Load())


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


def _parse_import(node: ast.Import):
    _local_modules = []
    # We don't want to override imports for outside packages
    for entry in node.names.copy():
        if entry.name.partition(".")[0] in LOCAL_MODULES:
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


# End AST helpers ##############


class ImportTransformer(ast.NodeTransformer):
    def __init__(self):
        self.insert_index = 0  # TODO hacky solution for __future__
        self.include_ivy_import = False

    def visit_Import(self, node):
        if isinstance(node, ast.Module):
            self.generic_visit(node)
            return node
        if isinstance(node, ast.Import):
            ret, should_impersonate = _parse_import(node)
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
                return _parse_absolute_fromimport(node)
            else:
                return _parse_relative_fromimport(node)

    def impersonate_import(self, tree: ast.Module):
        if self.include_ivy_import:
            tree.body.insert(
                self.insert_index,
                ast.ImportFrom(
                    module="ivy.utils.backend.backend_compiler",
                    names=[ast.alias(name="_ivy_fromimport")],
                    level=0,
                ),
            )
            tree.body.insert(
                self.insert_index,
                ast.ImportFrom(
                    module="ivy.utils.backend.backend_compiler",
                    names=[ast.alias(name="_ivy_absolute_import")],
                    level=0,
                ),
            )
        return tree


def _retrive_local_modules():
    ret = []
    wd = sys.path[0]
    for entry in os.scandir(wd):
        if entry.is_file():
            if entry.name.endswith(".py"):
                ret.append(entry.name[:-3])
                continue
        if entry.is_dir():
            if "__init__.py" in os.listdir(wd + "/" + entry.name):
                ret.append(entry.name)
    return ret


class IvyPathFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] not in _retrive_local_modules():
            # print("Global", fullname, "falling back to sys import")
            return None
        # We're local
        if path is None or path == "":
            path = [sys.path[0]]  # top level import --
        if "." in fullname:
            *parents, name = fullname.split(".")
        else:
            name = fullname
        for entry in path:
            if os.path.isdir(os.path.join(entry, name)):
                # this module has child modules
                filename = os.path.join(entry, name, "__init__.py")
                submodule_locations = [os.path.join(entry, name)]
            else:
                filename = os.path.join(entry, name + ".py")
                submodule_locations = None
            if not os.path.exists(filename):
                continue
            # print("Found Local", fullname)
            return spec_from_file_location(
                fullname,
                filename,
                loader=IvyLoader(filename),
                submodule_search_locations=submodule_locations,
            )
        # print("Couldn't find Local", fullname)
        return None


class IvyLoader(Loader):
    def __init__(self, filename):
        self.filename = filename

    def exec_module(self, module):
        with open(self.filename) as f:
            data = f.read()

        # print("Calling custom loader on ", module)
        ast_tree = parse(data)
        transformer = ImportTransformer()
        transformer.visit_ImportFrom(ast_tree)
        transformer.visit_Import(ast_tree)
        transformer.impersonate_import(ast_tree)
        ast.fix_missing_locations(ast_tree)
        try:
            exec(
                compile(ast_tree, filename=self.filename, mode="exec"), module.__dict__
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e


LOCAL_MODULES = _retrive_local_modules()
FINDER = IvyPathFinder()


def _clear_cache():
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
    spec = FINDER.find_spec(absolute_name, path)
    if spec is None:
        msg = f"No module named {absolute_name!r}"
        raise ModuleNotFoundError(msg, name=absolute_name)
    # print(spec, name)
    module = module_from_spec(spec)
    IMPORT_CACHE[absolute_name] = module
    spec.loader.exec_module(module)
    if path is not None:
        # Set reference to self in parent, if exist
        setattr(parent_module, child_name, module)
    return module


# We shouldn't be able to set the backend on a local Ivy
modules_to_remove = ["backend_handler"]


def with_backend(backend: str):
    sys.meta_path.insert(0, FINDER)
    ivy_pack = _ivy_import_module("ivy")
    ivy_pack._is_local = True
    backend_module = _ivy_import_module(
        ivy_pack.backend_handler._backend_dict[backend], ivy_pack.__package__
    )
    # TODO temporary
    if backend == "numpy":
        ivy_pack.set_default_device("cpu")
    elif backend == "jax":
        ivy_pack.set_global_attr("RNG", ivy_pack.functional.backends.jax.random.RNG)
    # We know for sure that the backend stack is empty, no need to do backend unsetting
    ivy_pack.backend_handler._set_backend_as_ivy(
        ivy_pack.__dict__.copy(), ivy_pack, backend_module
    )
    # Remove access to specific modules on local Ivy
    for module in modules_to_remove:
        for fn in inspect.getmembers(ivy_pack.__dict__[module], inspect.isfunction):
            if fn[1].__module__ != module:
                continue
            if hasattr(ivy_pack, fn[0]):
                del ivy_pack.__dict__[fn[0]]
        del ivy_pack.__dict__[module]
    ivy_pack.backend_stack.append(backend_module)
    sys.meta_path.remove(FINDER)
    _clear_cache()
    return ivy_pack
