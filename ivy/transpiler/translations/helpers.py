import gast
import importlib
from importlib.machinery import SourceFileLoader
import inspect
import os
import re
import subprocess
import sys
from typing import Any, Union, Optional, TYPE_CHECKING
from types import FunctionType

from ivy.transpiler.ast.source_gen import FileNameStrategy
from ivy.transpiler.core.object_like import BaseObjectLike
from ivy.transpiler.utils.naming_utils import NAME_GENERATOR
from ivy.transpiler.utils.source_utils import (
    get_new_output_dir_name,
    maybe_create_stateful_module,
    maybe_create_stateful_layers_module,
    sanitize_dir_name,
)

if TYPE_CHECKING:
    from ivy.transpiler.core.object_like import (
        FuncObjectLike,
        TypeObjectLike,
    )


def create_output_dir(
    object_like: Union[Any, "BaseObjectLike"],
    target: str,
    base_output_dir: str = "",
    use_legacy_dir_structure: bool = False,
) -> str:
    """
    Creates a unique output directory for the translated outputs of a given object and
    returns the path to this output directory.

    Args:
        object_like: The object for which the output directory is being created.
        target: The target framework (e.g., "torch_frontend", "ivy", "tensorflow").
        base_output_dir: The base output directory.
        use_legacy_dir_structure (bool, optional): If True, the function will create the output
                                                   directory using the legacy structure:
                                                   "<base_output_dir>.<object's_name>_output.run_<#>".
                                                   If False, it will use the new default structure:
                                                   "<base_output_dir>.<target>_outputs". Defaults to False.

    Returns:
        path: The path to the output directory.
    """

    root_dir = os.getcwd()
    if os.environ.get("UPDATE_S2S_CACHE", None) == "true":
        # write the translated files inside ivy_repo/compiler/_cache directory
        from ivy.transpiler.utils.cache_utils import ensure_cache_directory

        cache_dir = ensure_cache_directory()
        base_output_dir = os.path.join(cache_dir, base_output_dir)
    else:
        base_output_dir = os.path.join(root_dir, base_output_dir)

    # Determine the output directory name based on the structure type
    if use_legacy_dir_structure:
        base_output_dir = get_legacy_output_dir_name(object_like, base_output_dir)
    else:
        base_output_dir = get_new_output_dir_name(target, base_output_dir)

    # Create the output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Handle legacy directory structure for run-specific folders
    if use_legacy_dir_structure:
        subfolders = [
            folder
            for folder in os.listdir(base_output_dir)
            if os.path.isdir(os.path.join(base_output_dir, folder))
            and folder.startswith("run_")
        ]

        next_index = (
            0
            if not subfolders
            else max([int(subfolder.split("_")[1]) for subfolder in subfolders]) + 1
        )
        output_dir = os.path.join(base_output_dir, f"run_{next_index}")
    else:
        output_dir = base_output_dir

    is_windows = sys.platform.startswith("win")
    if is_windows:
        output_dir = os.path.normpath(output_dir)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create an empty __init__.py file in the new directory
    init_file = os.path.join(output_dir, "__init__.py")
    open(init_file, "a").close()

    # Maybe create a stateful module if target is a framework (e.g., TensorFlow)
    maybe_create_stateful_module(NAME_GENERATOR.target, output_dir)

    if os.environ.get("USE_NATIVE_FW_LAYERS", None) == "true":
        # Create stateful_layers.py for Keras layers if target is a framework (e.g., TensorFlow)
        maybe_create_stateful_layers_module(NAME_GENERATOR.target, output_dir)
    return output_dir


def get_legacy_output_dir_name(
    object_or_object_like: Union[
        FunctionType, type, "FuncObjectLike", "TypeObjectLike"
    ],
    base_output_dir: str,
) -> str:
    """
    Determines the legacy output directory name without creating it.

    Args:
        object_or_object_like: The object/object_like for which the legacy output directory name is being determined.
        base_output_dir: The base output directory.
    Returns:
        str: The legacy output directory name.
    """
    root_dir = os.getcwd()
    cls_ = NAME_GENERATOR.generate_name(object_or_object_like)
    base_output_dir = sanitize_dir_name(base_output_dir)
    return os.path.join(root_dir, base_output_dir, f"{cls_}_output")


def get_object_from_translated_directory(
    object_or_object_like: Union[
        FunctionType, type, "FuncObjectLike", "TypeObjectLike"
    ],
    translated_dir: str,
    base_output_dir: str,
    target: Optional[str] = "",
    reuse_existing: bool = True,
) -> Union[FunctionType, type, None]:
    """
    Checks if object exists in the translated directory.
    If it exists and reuse_existing is True, return the object.
    Otherwise, return None.

    Args:
        object_or_object_like: The object/object_like to check.
        translated_dir: The directory where the translated module is expected to be.
        base_output_dir: The base directory for the output files.
        target: The target framework (e.g., "torch_frontend", "ivy", "tensorflow").
        reuse_existing (bool): Flag to indicate whether to reuse the existing translation or not.

    Returns:
        The translated object if reuse_existing is True and the object exists; otherwise, None.
    """
    if not reuse_existing:
        return None

    if isinstance(object_or_object_like, BaseObjectLike):
        object_module = object_or_object_like.module
    elif isinstance(object_or_object_like, (FunctionType, type)):
        object_module = FileNameStrategy.infer_filename_from_module_name(
            object_or_object_like.__module__,
            as_module=True,
            base_output_dir=base_output_dir,
        )
    else:
        raise TypeError(f"Unsupported object type: {type(object_or_object_like)}")

    if target:
        NAME_GENERATOR.set_prefixes(target=target)
    translated_object_name = NAME_GENERATOR.generate_name(object_or_object_like)
    relative_path = translated_dir[len(os.getcwd()) + 1 :].replace(os.sep, ".")
    module_name = f"{relative_path}.{object_module}"
    try:
        # Construct the import statement
        import_statement = f"from {module_name} import {translated_object_name}"
        # Execute the import statement
        exec(import_statement)
        # Retrieve the object
        translated_object = eval(translated_object_name)
        return translated_object
    except Exception as e:
        return None


def safe_get_object_from_translated_directory(
    object_or_object_like: Union[
        FunctionType, type, "FuncObjectLike", "TypeObjectLike"
    ],
    translated_dir: str,
    base_output_dir: str,
    target: Optional[str] = "",
    reuse_existing: bool = True,
) -> Union[FunctionType, type, None]:
    """
    Safely retrieves an object from the translated directory by inspecting the module's source code.

    Unlike traditional approaches using `importlib` or `exec`, which execute the module during import, this
    method performs an AST (Abstract Syntax Tree) inspection of the source code to check if the object exists.
    This avoids unintended side effects such as triggering import-time code execution, which can arise when dynamically importing a module.

    Returns:
        The translated object if reuse_existing is True and the object exists; otherwise, None.
    """
    if not reuse_existing:
        return None

    if isinstance(object_or_object_like, BaseObjectLike):
        object_module = object_or_object_like.module
    elif isinstance(object_or_object_like, (FunctionType, type)):
        object_module = FileNameStrategy.infer_filename_from_module_name(
            object_or_object_like.__module__,
            as_module=True,
            base_output_dir=base_output_dir,
            target=target,
        )
    else:
        raise TypeError(f"Unsupported object type: {type(object)}")

    file_path = os.path.normpath(
        os.path.join(translated_dir, object_module.replace(".", os.sep) + ".py")
    )
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8", newline="\n") as f:
        source_code = f.read()

    object_name = NAME_GENERATOR.generate_name(object_or_object_like)
    source_ast = gast.parse(source_code)
    for node in gast.walk(source_ast):
        if isinstance(node, gast.FunctionDef):
            if node.name == object_name:
                return object_name
        elif isinstance(node, gast.ClassDef):
            if node.name == object_name:
                return object_name

    return None
