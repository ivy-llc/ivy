import ast
import os
import sys
import traceback
from ast import parse
from importlib.util import spec_from_file_location
from importlib.abc import Loader, MetaPathFinder


# AST helpers ##################


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
                    module="ivy.utils.backend._importlib",  # TODO remove str dependency
                    names=[ast.alias(name="_ivy_fromimport")],
                    level=0,
                ),
            )
            tree.body.insert(
                self.insert_index,
                ast.ImportFrom(
                    module="ivy.utils.backend._importlib",  # TODO remove str dependency
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
        if fullname.partition(".")[0] not in local_modules:
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


local_modules = _retrive_local_modules()
