import ast
import os
import sys
import traceback
from ast import parse
from importlib.util import spec_from_file_location
from importlib.abc import Loader, MetaPathFinder


# AST helpers ##################

# TODO add assertion to make sure module path exists
importlib_module_path = "ivy.utils._importlib"
importlib_abs_import_fn = "_absolute_import"
importlib_from_import_fn = "_from_import"
_unmodified_ivy_path = sys.modules["ivy"].__path__[0].rpartition("/")[0]
_compiled_modules_cache = {}


def _retrive_local_modules():
    ret = ["ivy"]  # TODO temporary hacky solution for finder
    # Get Ivy package root
    wd = sys.modules["ivy"].__path__[0]
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


def _parse_absolute_fromimport(node: ast.ImportFrom):
    # Not to override absolute imports to other packages
    if node.module.partition(".")[0] not in local_modules:
        return node
    to_import = []
    for entry in node.names:
        to_import.append((entry.name, entry.asname))
    # Return a function call
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id=importlib_from_import_fn, ctx=ast.Load()),
            args=[
                ast.Constant(value=node.module, kind=None),
                ast.Constant(value=None, kind=None),
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
            func=ast.Name(id=importlib_from_import_fn, ctx=ast.Load()),
            args=[
                ast.Constant(value=name, kind=None),
                ast.Name(id="__package__", ctx=ast.Load()),
                ast.Call(
                    func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]
                ),
                _create_list(to_import),
                ast.Constant(value=node.level, kind=None),
            ],
            keywords=[],
        ),
    )


def _create_list(elements):
    _elts = [ast.Constant(value=element, kind=None) for element in elements]
    return ast.List(elts=_elts, ctx=ast.Load())


def _create_assign_to_variable(target, value):
    return ast.Assign(
        targets=[ast.Name(id=target, ctx=ast.Store())],
        value=value,
    )


def _create_fromimport_call(name):
    return ast.Call(
        func=ast.Name(id=importlib_from_import_fn, ctx=ast.Load()),
        args=[
            ast.Constant(value=name, kind=None),
        ],
        keywords=[],
    )


def _parse_import(node: ast.Import):
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
                    func=ast.Name(id=importlib_abs_import_fn, ctx=ast.Load()),
                    args=[
                        ast.Constant(value=node.name, kind=None),
                        ast.Constant(value=node.asname, kind=None),
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
        ret, should_impersonate = _parse_import(node)
        if should_impersonate and not self.include_ivy_import:
            self.include_ivy_import = True
        return ret

    def visit_ImportFrom(self, node):
        self.include_ivy_import = True
        if node.level == 0:
            if node.module is not None and node.module == "__future__":
                self.insert_index = 1
            return _parse_absolute_fromimport(node)
        else:
            return _parse_relative_fromimport(node)

    def impersonate_import(self, tree: ast.Module, ivy_id=None):
        if self.include_ivy_import:
            if ivy_id is not None:
                ivy_import_node = ast.parse("import ivy").body[0]
                common_import_path = (
                    f"ivy.utils.backend.handler."
                    f"_compiled_backends_ids[{ivy_id}].utils._importlib"
                )
                abs_import_node = ast.parse(
                    f"{importlib_abs_import_fn} "
                    f"= {common_import_path}.{importlib_abs_import_fn}"
                ).body[0]
                from_import_node = ast.parse(
                    f"{importlib_from_import_fn} "
                    f"= {common_import_path}.{importlib_from_import_fn}"
                ).body[0]
                tree.body.insert(self.insert_index, from_import_node)
                tree.body.insert(self.insert_index, abs_import_node)
                tree.body.insert(self.insert_index, ivy_import_node)
            else:
                tree.body.insert(
                    self.insert_index,
                    ast.ImportFrom(
                        module=importlib_module_path,
                        names=[ast.alias(name=importlib_from_import_fn, asname=None)],
                        level=0,
                    ),
                )
                tree.body.insert(
                    self.insert_index,
                    ast.ImportFrom(
                        module=importlib_module_path,
                        names=[ast.alias(name=importlib_abs_import_fn, asname=None)],
                        level=0,
                    ),
                )
        return tree


class IvyPathFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] not in local_modules:
            return None
        # We're local
        if path is None or path == "":
            path = [_unmodified_ivy_path]
        if "." in fullname:
            *_, name = fullname.split(".")
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
            return spec_from_file_location(
                fullname,
                filename,
                loader=IvyLoader(filename),
                submodule_search_locations=submodule_locations,
            )
        return None


class IvyLoader(Loader):
    def __init__(self, filename):
        self.filename = filename

    def exec_module(self, module, ivy_id=None):
        if self.filename in _compiled_modules_cache:
            compiled_obj = _compiled_modules_cache[self.filename]
        else:
            with open(self.filename) as f:
                data = f.read()

            ast_tree = parse(data)
            transformer = ImportTransformer()
            transformer.visit(ast_tree)
            transformer.impersonate_import(ast_tree, ivy_id)
            ast.fix_missing_locations(ast_tree)
            compiled_obj = compile(ast_tree, filename=self.filename, mode="exec")
            _compiled_modules_cache[self.filename] = compiled_obj
        try:
            exec(compiled_obj, module.__dict__)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e
