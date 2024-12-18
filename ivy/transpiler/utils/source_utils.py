# global
from importlib.machinery import SourceFileLoader
import os
import inspect
import sys
import importlib
import subprocess
import re
import gast
from typing import Any, Union, Optional, TYPE_CHECKING
from types import FunctionType

# local
from ..translations.data.object_like import BaseObjectLike
from .naming_utils import NAME_GENERATOR
from .ast_utils import (
    FileNameStrategy,
)

if TYPE_CHECKING:
    from ..translations.data.object_like import (
        FuncObjectLike,
        TypeObjectLike,
    )


def _maybe_create_stateful_module(target: str, output_dir: str):
    if "frontend" in target or target in ("ivy", "numpy"):
        return

    # Retrieve source code from the corresponding stateful backend module in ivy
    import ivy

    source = "".join(inspect.findsource(ivy.NativeModule)[0])
    module_name = f"{NAME_GENERATOR.new_prefix}_stateful"
    stateful_file = os.path.join(output_dir, f"{module_name}.py")
    f = open(stateful_file, "w", encoding="utf-8", newline="\n")
    f.write(source)
    f.close()

    loader = SourceFileLoader(module_name, f.name)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules
    sys.modules[module.__name__] = module
    loader.exec_module(module)


def _maybe_create_stateful_layers_module(target: str, output_dir: str):
    from ivy.transpiler.transformations.transformers.native_layers_transformer import (
        KerasNativeLayers,
        FlaxNativeLayers,
    )

    if "frontend" in target or target in ("ivy", "numpy"):
        return

    if target == "tensorflow":
        if os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", "true") == "false":
            KerasNativeLayers = KerasNativeLayers.replace(
                "from .ivy.utils.decorator_utils import tensorflow_handle_transpose_in_input_and_output\n",
                "",
            )
            KerasNativeLayers = re.sub(
                r"@tensorflow_handle_transpose_in_input_and_output\n\s*",
                "",
                KerasNativeLayers,
            )

        source = KerasNativeLayers
        module_name = "tensorflow__stateful_layers"
    elif target == "jax":
        if os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", "true") == "false":
            FlaxNativeLayers = FlaxNativeLayers.replace(
                "from .ivy.utils.decorator_utils import jax_handle_transpose_in_input_and_output\n",
                "",
            )
            FlaxNativeLayers = re.sub(
                r"@jax_handle_transpose_in_input_and_output\n\s*",
                "",
                FlaxNativeLayers,
            )

        source = FlaxNativeLayers
        module_name = "jax__stateful_layers"
    stateful_file = os.path.join(output_dir, f"{module_name}.py")

    f = open(stateful_file, "w", encoding="utf-8", newline="\n")
    f.write(source)
    f.close()


def _reload_translated_modules(base_output_dir: str):
    """
    Deletes the translated modules from sys.modules, so they contain the latest changes.

    Reloads any modules containing `base_output_dir` in their name.
    """

    to_delete = []
    for key in sys.modules.keys():
        if base_output_dir in key:
            to_delete.append(key)
    for key in to_delete:
        del sys.modules[key]


def sanitize_dir_name(dir_name: str) -> str:
    """
    Sanitizes the directory name by removing illegal characters,
    leading dots, and adding a suffix if the name is 'ivy'.

    Args:
        dir_name: The directory name to sanitize.
    Returns:
        str: The sanitized directory name.
    """
    # Remove leading dots
    dir_name = dir_name.lstrip(".")

    # Replace illegal characters with underscores
    dir_name = re.sub(r'[<>:"/\\|?*\[\]]+', "_", dir_name)

    # Remove trailing underscores
    dir_name = dir_name.rstrip("_")

    # Add suffix if name is 'ivy'
    if dir_name == "ivy":
        dir_name = f"transpiled_{dir_name}"

    return dir_name


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


def get_new_output_dir_name(target: str, base_output_dir: str) -> str:
    """
    Determines the new output directory name without creating it.

    Args:
        target: The target framework.
        base_output_dir: The base output directory.
    Returns:
        str: The new output directory name.
    """
    root_dir = os.getcwd()
    return os.path.join(root_dir, base_output_dir, f"{target}_outputs")


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
    _maybe_create_stateful_module(NAME_GENERATOR.target, output_dir)

    if os.environ.get("USE_NATIVE_FW_LAYERS", None) == "true":
        # Create stateful_layers.py for Keras layers if target is a framework (e.g., TensorFlow)
        _maybe_create_stateful_layers_module(NAME_GENERATOR.target, output_dir)
    return output_dir


def format_code_with_ruff(file_path):
    # Run linting only on the final generated source code.
    subprocess.run(
        ["ruff", "check", "--fix-only", file_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    subprocess.run(
        ["ruff", "format", file_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )


def add_license(file_path: str):
    """
    Adds the license to the top of a translated file if it doesn't already exist.
    """
    file_path = os.path.normpath(file_path)
    with open(file_path, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # if the file is empty, do nothing
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


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


def determine_package_name(root: str, output_dir: str, base_output_dir: str) -> str:
    """
    Determine the correct package name based on the full path.

    Args:
    root (str): The root directory of the current file.
    output_dir (str): The output directory where the translated files are located.
    base_output_dir (str): The base of the output directory.

    Returns:
    str: The correctly formatted package name.
    """

    # Find the index of '.ivy' in the path
    if os.environ.get("UPDATE_S2S_CACHE", None) == "true":
        translated_outputs_index = output_dir.index(os.path.join("ivy", "compiler"))
    else:
        translated_outputs_index = output_dir.index(base_output_dir)

    # Get the full path from 'ivy_translated_outputs' to the root
    full_package_path = root[translated_outputs_index:]

    # Split the path and convert to package name format
    package_parts = full_package_path.split(os.sep)
    package_name = ".".join(package_parts)

    return package_name


def format_all_files_in_directory(
    output_dir, base_output_dir, translated_object_name, filename, logger, to_ignore
):
    """
    Formats all Python files in a directory and loads the modules.

    This function walks through a directory, and for each Python file,
    it formats the code and loads the module.
    Args:
        output_dir (str): The directory containing the Python files to format.
        base_output_dir (str): The root output directory.
        obj (object): An object from the module to be loaded.

    Returns:
        translated_object: The retrieved class from the loaded module.
    """
    from ivy.transpiler.utils.ast_utils import (
        reorder_module_objects,
        FileNameStrategy,
    )

    _reload_translated_modules(base_output_dir)

    def sort_files(file):
        return file != "__init__.py", file

    def sort_folders(folder_list):
        return sorted(folder_list, key=lambda x: (x != "ivy", x))

    # first reorder the module objects
    for root, dirs, files in os.walk(output_dir):

        dirs[:] = sort_folders(dirs)
        files.sort(key=sort_files)
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_to_lookup = file_path[len(output_dir) + 1 :].replace(os.sep, ".")
                if file_to_lookup in FileNameStrategy.FILES_MAP:
                    source_file = FileNameStrategy.FILES_MAP[file_to_lookup]
                    source_module = source_file[:-3]
                    try:
                        module = importlib.import_module(source_module)
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError(f"Module {source_module} not found")
                    if hasattr(module, "__file__"):
                        reorder_module_objects(
                            source_file_path=module.__file__,
                            target_file_path=file_path,
                            logger=logger,
                        )

    # first run linting and formatting on all files in the directory, and add license
    for root, _, files in os.walk(output_dir):
        dirs[:] = sort_folders(dirs)
        files.sort(key=sort_files)
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                add_license(file_path)
                format_code_with_ruff(file_path)

    # now load all the modules in the directory
    for root, _, files in os.walk(output_dir):
        dirs[:] = sort_folders(dirs)
        files.sort(key=sort_files)

        # Optionally disable __pycache__ generation
        original_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    module_name = os.path.basename(file_path[:-3])
                    if any(ignored in file for ignored in to_ignore):
                        continue

                    # Determine the correct package name
                    package_name = determine_package_name(
                        root, output_dir, base_output_dir=base_output_dir
                    )
                    if package_name:
                        module_name = f"{package_name}.{module_name}"

                    # Create and load the module
                    loader = SourceFileLoader(module_name, file_path)
                    module_spec = importlib.util.spec_from_loader(loader.name, loader)
                    if module_spec is None:
                        continue

                    module = importlib.util.module_from_spec(module_spec)
                    module.__package__ = package_name
                    try:
                        # Construct the import statement
                        import_statement = f"import {module_name}"
                        # Execute the import statement
                        exec(import_statement)
                        # Retrieve the object
                        module = eval(module_name)
                    except Exception as e:
                        raise ImportError(
                            f"Error loading module {module.__name__}: {e}"
                        )

                    # Add the module to sys.modules
                    sys.modules[module.__name__] = module
        finally:
            # Restore the original sys.dont_write_bytecode setting
            sys.dont_write_bytecode = original_dont_write_bytecode

    # Dynamically import the class from the module
    index = output_dir.index(base_output_dir)
    translated_dir_path = output_dir[index:].replace(os.sep, ".")
    full_module_name = f"{translated_dir_path}.{filename}"
    try:
        # Construct the import statement
        import_statement = f"from {full_module_name} import {translated_object_name}"
        # Execute the import statement
        exec(import_statement)
        # Retrieve the object
        translated_object = eval(translated_object_name)
    except Exception as e:
        raise ImportError(
            f"Failed to import {translated_object_name} from {full_module_name}: {e}"
        )

    return translated_object


def maybe_add_profiling_imports(output_dir: str):
    """
    Create a profiling.py file and import its namespace into the output directory files.
    """
    add_profiling_imports = os.environ.get("PROFILE_S2S", "false")
    if not add_profiling_imports == "true":
        return
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8", newline="\n") as f:
                    lines = f.readlines()

                idx = 0

                if len(lines) == 0:
                    lines.insert(0, "\n")

                while (
                    lines[idx].startswith("import")
                    or lines[idx].startswith("from")
                    or lines[idx].startswith("#")
                    or lines[idx] == "\n"
                ):
                    idx += 1
                    if idx == len(lines):
                        idx = 0
                        break

                lines.insert(idx, "\n")
                lines.insert(
                    idx,
                    "from ivy.transpiler import profiling_utils as profiling",
                )

                with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(lines)


def maybe_add_profiling_decorators(output_dir: str):
    add_profiling_decorators = os.environ.get("PROFILE_S2S", "false")
    if not add_profiling_decorators == "true":
        return
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8", newline="\n") as f:
                    lines = f.readlines()

                idx = len(lines) - 1

                if len(lines) == 0:
                    break

                while True:
                    if lines[idx].strip().startswith("def "):
                        whitespace = re.match(r"\s*", lines[idx]).group()
                        lines.insert(idx, "\n")
                        lines.insert(
                            idx, whitespace + "@profiling.profiling_logging_decorator"
                        )
                    idx -= 1
                    if idx == 0:
                        break

                with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(lines)
