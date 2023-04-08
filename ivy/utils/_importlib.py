import os
import sys
from pathlib import Path
from importlib.util import resolve_name, module_from_spec
from ivy.utils.backend import ast_helpers


import_cache = {}
path_hooks = []


class LocalIvyImporter:
    def __init__(self):
        self.finder = ast_helpers.IvyPathFinder()

    def __enter__(self):
        sys.meta_path.insert(0, self.finder)
        path_hooks.insert(0, self.finder)

    def __exit__(self, *exc):
        path_hooks.remove(self.finder)
        sys.meta_path.remove(self.finder)
        _clear_cache()


def _clear_cache():
    global import_cache
    import_cache = {}


def _from_import(name: str, package=None, mod_globals=None, from_list=(), level=0):
    """Handles absolute and relative from_import statement"""
    module_exist = name != ""
    name = "." * level + name
    module = _import_module(name, package)
    for entry_name, entry_asname in from_list:
        if entry_name == "*":
            if "__all__" in module.__dict__.keys():
                _all = {
                    k: v
                    for (k, v) in module.__dict__.items()
                    if k in module.__dict__["__all__"]
                }
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
            mod_globals[alias] = _import_module(in_name, package)
    return module


def _absolute_import(name: str, asname=None, mod_globals=None):
    """
    Handles absolute import statement
    :param name:
    :return:
    """
    if asname is None:
        _import_module(name)
        true_name = name.partition(".")[0]
        module = import_cache[true_name]
    else:
        true_name = asname
        module = _import_module(name)
    mod_globals[true_name] = module


def _get_all_modules(absolute_name: str):
    ivy_path = sys.modules["ivy"].__file__
    name_to_path = os.path.join(ivy_path, absolute_name.replace(".", os.path.sep))
    for root, dirs, files in os.walk(module_path):
        if root.endswith("__"):
            continue
        common = os.path.commonpath([root, module_path])
        package_full = root[len(common) :].replace(os.path.sep, ".")
        print(absolute_name + package_full)
        for name in files:
            if name.startswith("__") or not name.endswith(".py"):
                continue
            print(f"{absolute_name+package_full}.{name[:-3]}")
        for name in dirs:
            if name.startswith("__"):
                continue
            # print(name)


modules_to_skip = ["ivy.data_classes", "ivy.compiler"]

modules_to_exclude = ["ivy.data_classes.array.conversion"]

_full_modules_to_skip = _get_modules(modules_to_skip[0])
ivy_path_abs = Path(sys.modules["ivy"].__file__).parents[1]
_modules_to_skip = []


def _get_modules(absolute_name):
    name_to_path = absolute_name.replace(".", os.path.sep)
    module_path = os.path.join(ivy_path_abs, name_to_path)
    print(module_path)
    for root, dirs, files in os.walk(module_path):
        if root.endswith("__"):
            continue
        common = os.path.commonpath([root, module_path])
        package_full = root[len(common) :].replace(os.path.sep, ".")
        print(absolute_name + package_full)
        for name in files:
            if name.startswith("__") or not name.endswith(".py"):
                continue
            print(f"{absolute_name+package_full}.{name[:-3]}")
        for name in dirs:
            if name.startswith("__"):
                continue
            # print(name)


def _import_module(name, package=None):
    global import_cache
    absolute_name = resolve_name(name, package)
    try:
        return import_cache[absolute_name]
    except KeyError:
        pass

    path = None
    if "." in absolute_name:
        parent_name, _, child_name = absolute_name.rpartition(".")
        parent_module = _import_module(parent_name)
        path = parent_module.__spec__.submodule_search_locations
    for _module in modules_to_skip:
        if absolute_name.startswith(_module):
            print("Skipping", absolute_name)
            if path is not None:
                # Set reference to self in parent, if exist
                setattr(parent_module, child_name, sys.modules[absolute_name])
            return sys.modules[absolute_name]
    for finder in path_hooks:
        spec = finder.find_spec(absolute_name, path)
        if spec is not None:
            break
    else:
        msg = f"No module named {absolute_name!r}"
        raise ModuleNotFoundError(msg, name=absolute_name)
    module = module_from_spec(spec)
    import_cache[absolute_name] = module
    spec.loader.exec_module(module)
    if path is not None:
        # Set reference to self in parent, if exist
        setattr(parent_module, child_name, module)
    return module
