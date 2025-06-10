import importlib.util
import inspect
from pathlib import Path
import sys


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
