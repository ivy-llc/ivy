import os
import cv2
import re
import inspect

FRONTEND_PATH = "./ivy/functional/frontends/cv2/"


def _attr_from_name(module, element_name: str):
    return getattr(module, element_name, None)


def _is_uppercase(element: str) -> bool:
    return bool(re.compile(r"^[A-Z][A-Z_0-9]*$").match(element))


def _is_constant(module, element_name: str) -> bool:
    return isinstance(_attr_from_name(module, element_name), int)


def _is_class(module, element_name: str) -> bool:
    return inspect.isclass(_attr_from_name(module, element_name))


def _is_function(module, element_name: str) -> bool:
    return inspect.isfunction(_attr_from_name(module, element_name))


def _is_builtin(module, element_name: str) -> bool:
    return inspect.isbuiltin(_attr_from_name(module, element_name))


def _is_module(module, element_name: str) -> bool:
    return inspect.ismodule(_attr_from_name(module, element_name))


def _write_constants(file_path, module, constants: list[str]):
    with open(file_path, "a+") as file:
        file.write("# CONSTANTS \n")
        for constant_name in constants:
            value = str(getattr(module, constant_name, None))
            file.write(constant_name + " = " + value + "\n")
        file.write("\n\n")


def _write_builtins(file_path, module, builtins: list):
    with open(file_path, "a+") as file:
        file.write("# BUILTINS \n")
        for builtin in builtins:
            builtin_name = builtin.__name__
            file.write(f"#TODO: add {builtin_name}\n")
            doc = builtin.__doc__
            if doc is not None:
                commented_doc = ["#" + line for line in doc.splitlines()]
                commented_doc = "\n".join(commented_doc)
                file.write(commented_doc)
                file.write("\n")
            file.write("\n")
        file.write("\n\n")


def _write_functions(file_path, module, functions: list):
    with open(file_path, "a+") as file:
        file.write("# FUNCTIONS \n")
        for function in functions:
            function_name = function.__name__
            file.write(f"#TODO: add {function_name}\n")
            doc = function.__doc__
            if doc is not None:
                commented_doc = ["#" + line for line in doc.splitlines()]
                commented_doc = "\n".join(commented_doc)
                file.write(commented_doc)
                file.write("\n")
        file.write("\n\n")


def _module_file_name_from_module_name(module_name: str) -> str:
    return f"{module_name}.py"


def _module_name_from_module(module) -> str:
    return module.__name__.split(".")[-1]


def _unpack_module(module):
    module_name: str = _module_name_from_module(module)
    module_file_name: str = _module_file_name_from_module_name(module_name)

    names: list[str] = dir(module)

    constants: list[str] = []
    classes: dict = {}
    functions: list = []
    builtins_: list = []
    others: list = []

    for name in names:
        if _is_constant(module, name):
            constants.append(name)
            continue
        attr = getattr(module, name, None)

        if _is_class(module, name):
            classes[name] = {"doc": attr.__doc__, "mro": attr.__mro__}
            continue
        if _is_function(module, name):
            functions.append(attr)
            continue

        if _is_builtin(module, name):
            builtins_.append(attr)
            continue

        others.append(attr)

    module_file_path = os.path.join(FRONTEND_PATH, module_file_name)
    _write_constants(module_file_path, module, constants)
    _write_builtins(module_file_path, module, builtins_)
    _write_functions(module_file_path, module, functions)


if __name__ == "__main__":
    names: list[str] = dir(cv2)

    os.makedirs(FRONTEND_PATH, exist_ok=True)
    INIT_PATH = os.path.join(FRONTEND_PATH, "__init__.py")

    if not os.path.exists(INIT_PATH):
        with open(INIT_PATH, "w") as file:
            pass

    constants: list[str] = []
    classes: dict = {}
    functions: list = []
    builtins_: list = []
    modules: list = []
    others: list = []

    def _create_module_file(module_name: str):
        file_name: str = _module_file_name_from_module_name(module_name)
        file_path = os.path.join(FRONTEND_PATH, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file

    def _add_import_to_init(module_name: str):
        with open(INIT_PATH, "a") as file:
            file.write(f"from . import {module_name}\n")
            file.write(f"from .{module_name} import *\n")

    for name in names:
        if _is_constant(cv2, name):
            constants.append(name)
            continue
        attr = getattr(cv2, name, None)

        if _is_class(cv2, name):
            classes[name] = {"doc": attr.__doc__, "mro": attr.__mro__}
            continue
        if _is_function(cv2, name):
            functions.append(attr)
            continue

        if _is_builtin(cv2, name):
            builtins_.append(attr)
            continue

        if _is_module(cv2, name):
            if "cv2" in attr.__name__:
                modules.append(attr)
                module_name: str = _module_name_from_module(attr)
                _create_module_file(module_name)
                _add_import_to_init(module_name)
                _unpack_module(attr)
            continue

        others.append(attr)

    cv2
