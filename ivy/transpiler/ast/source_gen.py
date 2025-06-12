import gast
import os
from packaging.version import parse
import textwrap
from typing import Union, List, Dict, Set, Tuple, Optional, TYPE_CHECKING

import ivy.transpiler.ast.globals as glob
from .globals import (
    BACKEND_STANDARD_GLOBALS,
    FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE,
    IVY_STANDARD_GLOBALS_TARGET_TO_MODULE,
    MODULE_TO_ALIAS,
    MONKEY_PATCH_GLOBALS,
    TranslatedContext,
    TRANSLATED_OUTPUTS_SUBDIR,
)
from .analysis import get_translated_nodes
from .nodes import FromImportObj, ImportObj

from ivy.transpiler.core.object_like import BaseObjectLike
from ivy.transpiler.utils.api_utils import (
    get_native_module_str_from_backend,
    SUPPORTED_BACKENDS_PREFIX,
)
from ivy.transpiler.utils.ast_utils import ast_to_source_code, check_syntax
from ivy.transpiler.utils.cache_utils import (
    GlobalStatementCache,
    ImportStatementCache,
    ObjectLikeBytesToTranslatedObjectStringCache,
    EmittedSourceCache,
)
from ivy.transpiler.utils.naming_utils import NAME_GENERATOR
from ivy.transpiler.utils.type_utils import Types

if TYPE_CHECKING:
    from ivy.transpiler.core.global_like import (
        GlobalObjectLike,
    )
    from ivy.transpiler.core.object_like import (
        TypeObjectLike,
        FuncObjectLike,
    )


def _create_local_imports(import_statements: List[str]) -> List[gast.ImportFrom]:
    """
    This function creates AST nodes for import statements, specifically designed to be
    injected into a function or class scope to avoid circular import issues.

    Returns:
    list: A list of gast.ImportFrom nodes representing the local imports.
    """
    import_nodes = [
        gast.parse(import_statement).body[0]
        for import_statement in import_statements
        if "NestedSequence" not in import_statement
    ]

    return import_nodes


def _inject_builtin_imports(
    imports: Set[ImportObj],
    from_imports: Set[FromImportObj],
    import_statement_cache: ImportStatementCache,
    filename: str,
    old_imports: str,
    from_cache: bool = False,
) -> str:
    """
    Inject builtin imports into the sourcec code based on the imports present inside ast_transformer.imports
    and ast_transformer.from_imports

    This function processes import objects present in the ast_transformer, injecting them
    as source code into the current module depdending on whether its a regular import or a from import

    Returns:
    str: a combined string containing import statements

    Notes:
    - This function uses a global variable IMPORTS_ADDED to track added imports injected within the current module.
    """
    import_statements = []
    old_import_statements = old_imports.split("\n")
    for mod, asname in imports:
        if asname:
            import_stmt = f"import {mod} as {asname}"
        else:
            import_stmt = f"import {mod}"

        if import_stmt in old_import_statements:
            continue

        if from_cache:
            import_statements.append(import_stmt)
            if not import_statement_cache.exist(
                filename=filename, import_stmt=import_stmt
            ):
                import_statement_cache.cache(filename=filename, import_stmt=import_stmt)
        else:
            if not import_statement_cache.exist(
                filename=filename, import_stmt=import_stmt
            ):
                import_statements.append(import_stmt)
                import_statement_cache.cache(filename=filename, import_stmt=import_stmt)

    for mod, obj, asname in from_imports:
        if asname and obj != asname:
            import_stmt = f"from {mod} import {obj} as {asname}"
        else:
            import_stmt = f"from {mod} import {obj}"

        # If we are generating source code for a cached ACU from the preloaded cache, we
        # still need to make sure we inject the import statement to the current module
        if from_cache:
            import_statements.append(import_stmt)
            # If the current import doesn't exist in the ACU's cache, then cache it
            if not import_statement_cache.exist(
                filename=filename, import_stmt=import_stmt
            ):
                import_statement_cache.cache(filename=filename, import_stmt=import_stmt)
        # Otherwise, we are generating source code of a regular in-program ACU, in which case
        # we don't need to append the import statement unless it doesn't already exist in the import statement cache
        else:
            if not import_statement_cache.exist(
                filename=filename, import_stmt=import_stmt
            ):
                import_statements.append(import_stmt)
                import_statement_cache.cache(filename=filename, import_stmt=import_stmt)

    return "" if not import_statements else "\n".join(import_statements) + "\n\n"


def _inject_local_imports_in_function(
    import_nodes: List[gast.ImportFrom], func_name: str, ast_root: gast.AST
):
    """
    Inject import nodes into the Abstract Syntax Tree (AST) of a Python function.
    """

    # TODO: extract?
    class LocalNameCollector(gast.NodeVisitor):
        def __init__(self, current_name):
            self.current_name = current_name
            self.local_names = set()

        def visit_ClassDef(self, node):
            # Don't visit nested classees
            pass

        def visit_FunctionDef(self, node):
            # Don't visit nested functions
            if node.name != self.current_name:
                return
            self.generic_visit(node)

        def visit_Name(self, node):
            self.local_names.add(node.id)

    for node in gast.walk(ast_root):
        if isinstance(node, (gast.ClassDef, gast.FunctionDef)):

            # Collect all variable names used in this function's body
            collector = LocalNameCollector(current_name=node.name)
            collector.visit(node)

            # used_names = {n.id for n in gast.walk(node) if isinstance(n, gast.Name)}

            # Determine which import nodes are relevant for this function
            relevant_imports = [
                n for n in import_nodes if n.names[0].name in collector.local_names
            ]

            # If there are relevant imports, inject them at the start of the function/class body
            if relevant_imports:
                node.body = relevant_imports + node.body


def _inject_local_imports_in_class(
    import_nodes: List[gast.ImportFrom], ast_root: gast.AST
):
    """
    Inject import nodes into the Abstract Syntax Tree (AST) of a Python class,
    but specifically into the methods where the imports are referenced.
    """

    for node in gast.walk(ast_root):
        if isinstance(node, gast.FunctionDef):
            # Collect all variable names used in this function's body
            used_names = {n.id for n in gast.walk(node) if isinstance(n, gast.Name)}

            # Determine which import nodes are relevant for this function
            relevant_imports = [
                n for n in import_nodes if n.names[0].name in used_names
            ]

            # If there are relevant imports, inject them at the start of the function body
            if relevant_imports:
                node.body = relevant_imports + node.body


def _inject_standard_globals(
    object_like: Union["FuncObjectLike", "TypeObjectLike"],
    source: str,
    target: str,
    output_dir: str,
    global_statement_cache: GlobalStatementCache,
    filename: str,
) -> str:
    """
    Inject standard global variables into the AST based on the target backend.

    This function adds standard global variables to the AST, particularly for
    supported backends.

    Returns:
    str: A string containing the standard global variables to be injected, or an empty string if not needed.
    """
    standard_globals = []

    # TODO: Make this extensible
    if target in BACKEND_STANDARD_GLOBALS:
        for standard_global in BACKEND_STANDARD_GLOBALS[target]:
            standard_globals.append(standard_global)

    return "".join(standard_globals)


def _inject_standard_imports(
    object_like: Union["FuncObjectLike", "TypeObjectLike"],
    target: str,
    import_statement_cache: ImportStatementCache,
    old_imports: str,
    filename: str,
) -> str:
    old_import_statements = old_imports.split("\n")
    if target == "torch_frontend":
        import_statements = [
            "import ivy.functional.frontends.torch as torch",
            "import ivy.functional.frontends.torch.nn as nn",
            "import ivy",
            "import numpy as np",
        ]
    elif target == "ivy":
        import_statements = [
            "import ivy",
            "from collections import OrderedDict",
            "import threading",
            "import inspect",
            "import numpy as np",
        ]
    else:
        import_statements = [
            f"import {target}",
            (
                f"import {target} as {MODULE_TO_ALIAS[target]}"
                if target in MODULE_TO_ALIAS
                else ""
            ),
            "from collections import OrderedDict",
            "import threading",
            "import inspect",
            "import numpy as np",
        ]
        if target == "jax":
            try:
                import flax

                if parse(flax.__version__) >= parse("0.10.0"):
                    ModulePath = "flax.nnx.module"
                else:
                    ModulePath = "flax.nnx.nnx.module"
            except ImportError:
                raise ImportError(
                    "flax is not installed. Please install flax to use ivy.transpile with the target as jax."
                )
            import_statements += [
                "import jaxlib",
                "import jax.numpy as jnp",
                "import flax.nnx as nnx",
                f"from {ModulePath} import ModuleMeta",
            ]

    import_statements = [
        imp for imp in import_statements if imp not in old_import_statements
    ]
    import_statement_strings = "\n".join(import_statements) + "\n\n"
    if not import_statement_cache.exist(
        filename=filename, import_stmt=import_statement_strings
    ):
        import_statement_cache.cache(
            filename=filename, import_stmt=import_statement_strings
        )
        return import_statement_strings
    return ""


def _inject_module_dependencies(
    translated_strings: Dict[str, TranslatedContext],
    target: str,
    object_like_bytes_to_translated_object_str_cache: ObjectLikeBytesToTranslatedObjectStringCache,
    import_statement_cache: ImportStatementCache,
    old_imports: str,
    circular_reference_object_likes: Set[Union["FuncObjectLike", "TypeObjectLike"]],
    object_like: Union["FuncObjectLike", "TypeObjectLike"],
    current_object_name: str,
    current_filename: str,
    base_output_dir: str,
    ast_root: gast.AST = None,
) -> str:
    """
    Inject necessary module dependencies into the target module as import statements.

    This function analyzes the dependencies between objects in the translated strings and the
    current object to determine the appropriate imports needed. It handles cases involving
    circular dependencies and ensures that imports are added correctly either at the module
    or local level to prevent circular imports.

    Parameters
    ----------
    translated_strings : Dict[str, TranslatedContext]
        A dictionary mapping translated object names to their context.
    target : str
        The target module where dependencies are injected.
    object_like_bytes_to_translated_object_str_cache : ObjectLikeBytesToTranslatedObjectStringCache
        Cache containing mappings between object bytes and their translated string names.
    import_statement_cache : ImportStatementCache
        Cache for managing import statements and avoiding duplicate imports.
    old_imports : str
        Existing import statements in the current module, used to avoid redundant imports.
    circular_reference_object_likes : Set[Union["FuncObjectLike", "TypeObjectLike"]]
        Set of objects involved in circular reference situations that require special import handling.
    object_like : Union["FuncObjectLike", "TypeObjectLike"]
        The current object being analyzed, which may require new import statements.
    current_object_name : str
        Name of the current object for which dependencies are being injected.
    current_filename : str
        Filename of the current module where dependencies are injected.
    base_output_dir : str
        Base directory for the output files.
    ast_root : gast.AST, optional
        The root of the AST (Abstract Syntax Tree) for injecting local imports into function
        or class bodies.

    Returns
    -------
    str
        A string containing the necessary module-level import statements. Returns an empty
        string if no imports are required.

    """
    module_imports = []
    local_imports = []
    local_circular_imports = []
    module_circular_imports = []
    old_import_statements = old_imports.split("\n")

    # create a reverse map of the object_like_bytes_to_translated_object_str cache
    translated_object_str_to_object_like_cache = dict(
        map(reversed, object_like_bytes_to_translated_object_str_cache._cache.items())
    )

    for translated_name, ctx in translated_strings.items():
        translated_obj_like_bytes = translated_object_str_to_object_like_cache.get(
            translated_name, None
        )
        if translated_obj_like_bytes:
            translated_obj_like = BaseObjectLike.loads(translated_obj_like_bytes)
            curr_obj_like = object_like
            if curr_obj_like.filename == translated_obj_like.filename:
                # Case 1: Same Module - No import needed
                pass
            else:
                # Case 2: Different Module - add  Import
                module_name = translated_obj_like.filename[:-3]
                """from <mod> import <object>"""
                if _validate_from_import(
                    from_mod=module_name,
                    from_obj=translated_name,
                    current_module_name=current_filename[:-3],
                    current_object_name=current_object_name,
                ):
                    import_stmt = create_relative_import_statement(
                        from_mod=module_name,
                        from_obj=translated_name,
                        current_module_name=current_filename[:-3],
                    )
                    is_compile_time_object = ctx != TranslatedContext.VARIABLE
                    # if a compile time object (eg: decorator, type spec etc.): add as local import
                    if not is_compile_time_object:
                        local_imports.append(import_stmt)
                    else:
                        # else add as module import
                        if (
                            not import_statement_cache.exist(
                                filename=current_filename, import_stmt=import_stmt
                            )
                            and import_stmt not in old_import_statements
                        ):
                            module_imports.append(import_stmt)
                            import_statement_cache.cache(
                                filename=current_filename, import_stmt=import_stmt
                            )

    module_imports.sort()

    if local_imports:
        import_nodes = _create_local_imports(local_imports)
        if object_like.type == Types.FunctionType:
            func_name = NAME_GENERATOR.get_name(object_like)
            _inject_local_imports_in_function(import_nodes, func_name, ast_root)
        else:
            _inject_local_imports_in_class(import_nodes, ast_root)

    """
    Special Case: handling circular references

    Example:

    ``` # module <A.py>       
    from .B import B
    class A(B):
        def __init__(self):
            super().__init__()
        
        def foo(self):
            pass
    ```
    
    ``` # module <B.py>       
    from .A import A
    class B():
        def __init__(self):
            super().__init__()

        def foo(self):
            if isinstance(self, A): # circular reference
                self.foo()
    ```
    # inside B.py, we cannot have an import statement for A. This will lead to a 
    # circular import issue. To resolve this, we will have to add a local import
    # inside the class B. Hence, class B will become 
    class B(A):
        def __init__(self):
            super().__init__()
        
        def foo(self):
            from .A import A
            if isinstance(self, A): 
                self.foo()
    """
    if circular_reference_object_likes:
        for obj_like in circular_reference_object_likes:
            translated_name = NAME_GENERATOR.generate_name(obj_like)
            parent_module = FileNameStrategy.infer_filename_from_object_like(
                obj_like,
                target,
                as_module=True,
                base_output_dir=base_output_dir,
            )
            """from <mod> import <object>"""
            if _validate_from_import(
                from_mod=parent_module,
                from_obj=translated_name,
                current_module_name=current_filename[:-3],
                current_object_name=current_object_name,
            ):
                import_stmt = create_relative_import_statement(
                    from_mod=parent_module,
                    from_obj=translated_name,
                    current_module_name=current_filename[:-3],
                )
                is_compile_time_object = obj_like.ctx != TranslatedContext.VARIABLE
                if is_compile_time_object:
                    if (
                        not import_statement_cache.exist(
                            filename=current_filename, import_stmt=import_stmt
                        )
                        and not import_stmt in old_import_statements
                    ):
                        module_circular_imports.append(import_stmt)
                        import_statement_cache.cache(
                            filename=current_filename, import_stmt=import_stmt
                        )
                else:
                    local_circular_imports.append(import_stmt)

        local_circular_import_nodes = _create_local_imports(local_circular_imports)
        if obj_like.type == Types.FunctionType:
            func_name = NAME_GENERATOR.get_name(obj_like)
            _inject_local_imports_in_function(
                local_circular_import_nodes, current_object_name, ast_root
            )
        else:
            _inject_local_imports_in_class(local_circular_import_nodes, ast_root)

    imports = module_imports + module_circular_imports
    return "" if not imports else "\n".join(imports) + "\n\n"


def _maybe_inject_frontend_standard_globals(
    source: str,
    target: str,
    output_dir: str,
    base_output_dir: str,
    global_statement_cache: GlobalStatementCache,
) -> None:
    """Inject standard frontend global variables into the appropriate frontend `__init__` files."""

    # Populate the globals first
    _maybe_populate_frontend_standard_globals(source=source, target=target)

    if (
        target in SUPPORTED_BACKENDS_PREFIX
        or source == "torch_frontend"
        and target == "ivy"
    ):
        if target in SUPPORTED_BACKENDS_PREFIX:
            # ivy standard globals
            import ivy

            for target_str, assign_str in glob.IVY_STANDARD_GLOBALS.items():
                ivy_standard_global = f"\n{target_str} = {assign_str}\n"

                # inject global into the correct module
                module = IVY_STANDARD_GLOBALS_TARGET_TO_MODULE[target_str]
                module_path = module.replace(".", os.sep)
                file = os.path.join(output_dir, f"{module_path}.py")

                if not global_statement_cache.exist(
                    filename=file, glob_stmt=ivy_standard_global
                ):
                    root_module, _ = module_path.rsplit(os.sep, 1)
                    dir = os.path.join(output_dir, root_module)
                    os.makedirs(dir, exist_ok=True)
                    is_new_file = not os.path.exists(file)
                    mode = "a" if not is_new_file else "w"
                    with open(file, mode, encoding="utf-8", newline="\n") as f:
                        f.write(ivy_standard_global)

                    # add a mapping for this file so that we can reorder objects present within it.
                    start_index = file.index(base_output_dir)
                    file_key = file[start_index:].replace(os.sep, ".")
                    py_filename = module.rsplit(".", 1)[0]
                    file_key_for_files_map = f"{py_filename}.py"
                    FileNameStrategy.FILES_MAP[file_key] = file_key_for_files_map

                    global_statement_cache.cache(
                        filename=file, glob_stmt=ivy_standard_global
                    )

        # frontend standard globals
        import ivy.functional.frontends.torch
        import ivy.functional.frontends.numpy

        for target_str, assign_str in glob.FRONTEND_STANDARD_GLOBALS.items():
            frontend_standard_global = f"\n{target_str} = {assign_str}\n"

            # inject global into the correct module
            module = FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE[target_str]
            module_path = module.replace(".", os.sep)
            file = os.path.join(output_dir, f"{module_path}.py")

            if not global_statement_cache.exist(
                filename=file, glob_stmt=frontend_standard_global
            ):
                root_module, _ = module_path.rsplit(os.sep, 1)
                dir = os.path.join(output_dir, root_module)
                # only inject numpy globals if "ivy/functional/frontends/numpy" dir exists
                if "numpy" not in dir or "numpy" in dir and os.path.exists(dir):
                    os.makedirs(dir, exist_ok=True)
                    is_new_file = not os.path.exists(file)
                    mode = "a" if not is_new_file else "w"
                    with open(file, mode, encoding="utf-8", newline="\n") as f:
                        f.write(frontend_standard_global)

                # add a mapping for this file so that we can reorder objects present within it.
                start_index = file.index(base_output_dir)
                file_key = file[start_index:].replace(os.sep, ".")
                py_filename = module.rsplit(".", 1)[0]
                file_key_for_files_map = f"{py_filename}.py"
                FileNameStrategy.FILES_MAP[file_key] = file_key_for_files_map

                global_statement_cache.cache(
                    filename=file, glob_stmt=frontend_standard_global
                )


def _maybe_inject_stateful_import(
    target: str,
    ast_root: gast.AST,
    filename: str,
    import_statement_cache: ImportStatementCache,
    old_imports: str,
    inject_import: bool = False,
    object_like: Union["FuncObjectLike", "TypeObjectLike"] = None,
) -> str:
    """
    Inject a stateful import into the current module based on the target backend.
    An example of a stateful import when target='tensorflow' is:
    "from tensorflow_stateful import Layer as tensorflow_keras_Layer"

    Returns:
    str: A string containing the stateful import to be injected, or an empty string if not needed.
    """
    if "frontend" in target or target in ("ivy", "numpy"):
        return ""

    if not inject_import:
        return ""

    old_import_statements = old_imports.split("\n")
    stateful_mod = f"{NAME_GENERATOR.new_prefix}_stateful"
    stateful_cls_name = get_native_module_str_from_backend(
        backend_str=target,
        is_root_obj=object_like.is_root_obj,
        depth=object_like.depth,
    )
    name = stateful_cls_name.split(".")[-1]
    alias = stateful_cls_name.replace(".", "_")
    alias_suffix = "_".join(alias.split("_")[:2])

    # Modify the import here to correctly represent the base class, needed in case
    # the cached object like was a root obj (tensorflow_keras_Model) but the
    # depth for the retrieved object like != 0 or not is_root_obj (tensorflow_keras_Layer)
    # or vice versa, resulting in a mismatch between the import and the base class
    if isinstance(ast_root, gast.Module) and ast_root.body:
        class_node = ast_root.body[0]
        if isinstance(class_node, gast.ClassDef):
            for base in class_node.bases:
                base_name = ast_to_source_code(base).strip()
                if alias_suffix in base_name and alias != base_name:
                    alias = base_name
                    break

    import_statement = create_relative_import_statement(
        from_mod=stateful_mod,
        from_obj=name,
        current_module_name=filename[:-3],
        asname=alias,
    )

    if (
        not import_statement_cache.exist(
            filename=filename, import_stmt=import_statement
        )
        and import_statement not in old_import_statements
    ):
        import_statement_cache.cache(filename=filename, import_stmt=import_statement)
        return "\n" + import_statement + "\n"
    return ""


def _maybe_populate_frontend_standard_globals(source: str, target: str) -> None:
    if (
        target in SUPPORTED_BACKENDS_PREFIX
        or source == "torch_frontend"
        and target == "ivy"
    ):
        if target in SUPPORTED_BACKENDS_PREFIX:
            # ivy standard globals
            import ivy

            glob.IVY_STANDARD_GLOBALS["promotion_table"] = repr(ivy.promotion_table)
            glob.IVY_STANDARD_GLOBALS["array_api_promotion_table"] = repr(
                ivy.array_api_promotion_table
            )

    # frontend standard globals
    import ivy.functional.frontends.torch
    import ivy.functional.frontends.numpy

    glob.FRONTEND_STANDARD_GLOBALS["torch_promotion_table"] = repr(
        ivy.functional.frontends.torch.torch_promotion_table
    )
    glob.FRONTEND_STANDARD_GLOBALS["numpy_promotion_table"] = repr(
        ivy.functional.frontends.numpy.numpy_promotion_table
    )
    glob.FRONTEND_STANDARD_GLOBALS["numpy_str_to_type_table"] = repr(
        ivy.functional.frontends.numpy.numpy_str_to_type_table
    )
    # TODO: Add support translating these globals which contain custom objects from the numpy frontend
    # FRONTEND_STANDARD_GLOBALS["numpy_scalar_to_dtype"] = repr(ivy.functional.frontends.numpy.numpy_scalar_to_dtype)
    # FRONTEND_STANDARD_GLOBALS["numpy_dtype_to_scalar"] = repr(ivy.functional.frontends.numpy.numpy_dtype_to_scalar)
    glob.FRONTEND_STANDARD_GLOBALS["numpy_casting_rules"] = repr(
        ivy.functional.frontends.numpy.numpy_casting_rules
    )


def _sort_statements(statements):
    """
    sorts the imports statements in ascending order
    """
    # Split the statements into a list
    statements_list = statements.split("\n")
    # Sort the statements
    statements_list.sort()
    # Combine all the statements
    return "\n".join(statements_list)


def _validate_from_import(
    from_mod: str,
    from_obj: str,
    current_module_name: str,
    current_object_name: str,
):
    """
    Validate whether an import statement from a module and object is valid in the current context.

    This function checks whether the specified module and object to be imported are distinct from
    the current module and object. It ensures that the module and object are not the same as the
    current module or object, which could result in circular imports or redundant imports.

    Parameters
    ----------
    from_mod : str
        The module from which the object is being imported. This should be the module name without a `.py` extension.
    from_obj : str
        The name of the object being imported (e.g., a class or function).
    current_module_name : str
        The name of the current module where the validation is being performed, without the `.py` extension.
    current_object_name : str
        The name of the current object (e.g., a class or function) in the current module.

    Returns
    -------
    bool
        True if the module and object to be imported are valid (i.e., they differ from the current module and object),
        otherwise False.

    Notes
    -----
    - The function uses two guards:
        1. The module (`from_mod`) should not be the same as the `current_module_name`.
        2. The object (`from_obj`) should not be the same as the `current_object_name`.
    - This helps avoid importing the same module or object in the current context, which could lead to circular dependencies.

    Examples
    --------
    Validating an import from a different module and object:

    >>> _validate_from_import("module_A", "ClassA", "module_B", "ClassB")
    True

    Invalidating an import from the same module:

    >>> _validate_from_import("module_A", "ClassA", "module_A", "ClassB")
    False

    Invalidating an import of the same object:

    >>> _validate_from_import("module_A", "ClassA", "module_A", "ClassA")
    False
    """
    assert not from_mod.endswith(
        ".py"
    ), f"from_mod should not end with .py. got {from_mod}"
    assert not current_module_name.endswith(
        ".py"
    ), f"current_module_name should not end with .py. got {current_module_name}"
    is_module_valid = lambda mod: mod != current_module_name
    is_imported_obj_valid = lambda obj_name: obj_name != current_object_name
    return is_module_valid(from_mod) and is_imported_obj_valid(from_obj)


class FileNameStrategy:
    FILES_MAP = {
        "ivy.__init__.py": "ivy.py",
        "ivy.functional.frontends.torch.__init__.py": "ivy.functional.frontends.torch.py",
        "ivy.functional.frontends.tensorflow.__init__.py": "ivy.functional.frontends.tensorflow.py",
        "ivy.functional.frontends.jax.__init__.py": "ivy.functional.frontends.jax.py",
        "ivy.functional.frontends.numpy.__init__.py": "ivy.functional.frontends.numpy.py",
        "ivy.functional.backends.torch.__init__.py": "ivy.functional.backends.torch.py",
        "ivy.functional.backends.tensorflow.__init__.py": "ivy.functional.backends.tensorflow.py",
        "ivy.functional.backends.jax.__init__.py": "ivy.functional.backends.jax.py",
        "ivy.functional.backends.numpy.__init__.py": "ivy.functional.backends.numpy.py",
    }

    @staticmethod
    def infer_filename_from_object_like(
        object_like: Union["FuncObjectLike", "TypeObjectLike"],
        target: str,
        base_output_dir: str,
        as_module: bool = False,
    ):
        """
        Infer the file name from an object-like structure's module name.
        """

        # pattern1: base_output_dir.<...>.run_<..?>.
        # pattern2: base_output_dir.ivy_outputs.<...>
        obj_module = object_like.module

        assert not obj_module.endswith(
            ".py"
        ), f"object_like.module must not end with .py, got {object_like.module}"
        parts = obj_module.lstrip(".").split(".")
        if parts[0] == base_output_dir and parts[2].startswith("run_"):
            # strip off the first 3 parts and this is the new module name
            new_module = ".".join(parts[3:])
        elif parts[0] == base_output_dir and parts[1] in (TRANSLATED_OUTPUTS_SUBDIR):
            # strip off the first 2 parts and this is the new module name
            new_module = ".".join(parts[2:])
        # add a mapping to the file name
        else:
            # module is the same as object_like.module
            new_module = ".".join(parts)

        raw_filename = obj_module + ".py"
        if raw_filename not in FileNameStrategy.FILES_MAP:
            FileNameStrategy.FILES_MAP[raw_filename] = raw_filename.replace(
                ".__init__", ""
            )
        if as_module:
            return new_module
        return new_module + ".py"

    @staticmethod
    def infer_filename_from_module_name(
        module_name: str,
        base_output_dir: str,
        as_module: bool = False,
    ):
        """
        Infer the file name from a module name.
        """

        assert not module_name.endswith(
            ".py"
        ), f"module_name must not end with .py. got {module_name}"
        # pattern1: base_output_dir.<...>.run_<..?>.
        # pattern2: base_output_dir.ivy_outputs.<...>
        parts = module_name.lstrip(".").split(".")
        if parts[0] == base_output_dir and parts[2].startswith("run_"):
            # strip off the first 3 parts and this is the new module name
            new_module = ".".join(parts[3:])
        elif parts[0] == base_output_dir and parts[1] in (TRANSLATED_OUTPUTS_SUBDIR):
            # strip off the first 2 parts and this is the new module name
            new_module = ".".join(parts[2:])
        # add a mapping to the file name
        else:
            # module is the same as object_like.module
            new_module = ".".join(parts)

        raw_filename = module_name + ".py"
        if raw_filename not in FileNameStrategy.FILES_MAP:
            FileNameStrategy.FILES_MAP[raw_filename] = raw_filename.replace(
                ".__init__", ""
            )

        if as_module:
            return new_module
        return new_module + ".py"

    @staticmethod
    def create_module_structure(output_dir: str, module_path: str, target: str) -> str:
        """
        Create the directory structure for the given module path.

        Args:
        output_dir (str): The base output directory.
        module_path (str): The full module path.

        Returns:
        str: The full path to the directory where the module should be created.
        """
        # Remove the file extension if present
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        # maybe add monkey patching globals to the root dir's __init__.py
        file = os.path.join(output_dir, "__init__.py")
        if target in MONKEY_PATCH_GLOBALS and os.path.getsize(file) == 0:
            with open(file, "w", encoding="utf-8", newline="\n") as f:
                file_content = textwrap.dedent(MONKEY_PATCH_GLOBALS[target])
                f.write(file_content)

        # Split the module path into parts
        parts = module_path.split(".")

        # If it's a single file module, return the output_dir
        if len(parts) == 1:
            return os.path.join(output_dir, parts[0]) + ".py"

        # Create the directory structure
        current_dir = output_dir
        for part in parts[:-1]:
            current_dir = os.path.join(current_dir, part)
            os.makedirs(current_dir, exist_ok=True)

            # Create __init__.py file
            init_file = os.path.join(current_dir, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w", encoding="utf-8", newline="\n") as f:
                    pass  # Create an empty file

        return os.path.join(current_dir, parts[-1]) + ".py"


def add_global_assignments_and_create_imports(
    ast_root: gast.AST,
    object_like: Union["FuncObjectLike", "TypeObjectLike"],
    global_objects: List["GlobalObjectLike"],
    global_statement_cache: GlobalStatementCache,
    import_statement_cache: ImportStatementCache,
    imports: List[ImportObj],
    from_imports: List[FromImportObj],
    old_imports: str,
    current_filename: str,
    current_object_name: str,
    output_dir: str,
    target: str,
    from_cache: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Add global assignments and create necessary import statements in the module.

    This function processes global objects, generating appropriate global assignment statements
    and creating import statements for any dependencies. It handles importing necessary modules
    or objects at the global or local scope, ensuring that all dependencies are properly handled
    and avoiding duplicate imports or circular dependencies.

    Parameters
    ----------
    ast_root : gast.AST
        The root of the AST (Abstract Syntax Tree) of the current object being processed.
    object_like : Union["FuncObjectLike", "TypeObjectLike"]
        The current object being processed, which could be a function or a class.
    global_objects : List["GlobalObjectLike"]
        A list of global objects that require assignment and possibly imports.
    global_statement_cache : GlobalStatementCache
        A cache to track previously added global assignments to avoid duplication.
    import_statement_cache : ImportStatementCache
        A cache to track and avoid duplicate import statements.
    imports : List[ImportObj]
        A list of 'import ...' import objects to inject.
    from_imports : List[FromImportObj]
        A list of 'from ... import ...' style imports to inject.
    old_imports : str
        The current import statements in the module, used to avoid adding duplicates.
    current_filename : str
        The name of the file currently being processed.
    current_object_name : str
        The name of the current object (function or class) being processed.
    output_dir : str
        The directory where output files are written, used when creating new files or updating existing ones.
    target : str
        The target directory or module where dependencies should be injected.
    from_cache : bool, optional
        Whether the global objects are being loaded from a cache, by default False.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing:
        - A list of global assignment strings to be added to the module.
        - A list of module-level import statement strings to be added to the module.

    Notes
    -----
    - Global assignments are added for any objects not yet cached.
    - Dependencies for global objects are imported either at the module or local level depending on the context.
    - For global objects in different modules, necessary import statements are generated and injected into the target module.
    - Circular dependencies are handled by injecting imports into function or class bodies as local imports when necessary.

    """
    global_assignment_strings = []
    module_import_statement_strings = []
    local_import_statement_strings = []
    old_import_statements = old_imports.split("\n")

    for glob in global_objects:
        if glob.global_filename not in FileNameStrategy.FILES_MAP:
            FileNameStrategy.FILES_MAP[glob.global_filename] = (
                glob.global_filename.replace(".__init__", "")
            )
        # Task 1: Add global assignment in its corresponding module (ie: glob.filename)
        if glob.global_filename != current_filename:
            imports_to_inject = (
                _inject_standard_imports(
                    object_like,
                    target=target,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=glob.global_filename,
                )
                + _inject_builtin_imports(
                    imports=imports,
                    from_imports=from_imports,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=glob.global_filename,
                    from_cache=from_cache,
                )
                + _maybe_inject_stateful_import(
                    target=target,
                    inject_import=True,
                    object_like=object_like,
                    ast_root=ast_root,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=glob.global_filename,
                )
            )

            # check if the global statmenthas already been added inside global_filename
            if not global_statement_cache.exist(
                filename=glob.global_filename, glob_stmt=glob.assignment_str
            ):
                global_assignment_string = glob.assignment_str
                # add the global statement to the cache
                global_statement_cache.cache(
                    filename=glob.global_filename, glob_stmt=glob.assignment_str
                )
            else:
                global_assignment_string = ""

            fullpath = FileNameStrategy.create_module_structure(
                output_dir, glob.global_filename, target
            )
            file_path = fullpath
            is_new_file = not os.path.exists(file_path)
            new_source = imports_to_inject + "\n" + global_assignment_string + "\n"
            if new_source.strip():  # check if the source is not empty
                if is_new_file:
                    # check for any syntax errors
                    check_syntax(new_source)
                    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
                        file.write(new_source + "\n")
                else:
                    with open(file_path, "r", encoding="utf-8", newline="\n") as f:
                        # read the existing content
                        old_source = f.read()

                    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
                        old_imports_str, old_source_code_and_globals = (
                            _split_imports_globals_and_code(source=old_source)
                        )
                        combined_imports = _sort_statements(
                            old_imports_str + imports_to_inject
                        )
                        combined_source = (
                            combined_imports
                            + "\n"
                            + old_source_code_and_globals
                            + "\n"
                            + global_assignment_string
                            + "\n"
                        )
                        # check for any syntax errors
                        check_syntax(combined_source)

                        file.seek(0, 0)
                        file.write(combined_source)

            # Task 2: Create import statements
            module_name = glob.global_filename[:-3]
            if _validate_from_import(
                from_mod=module_name,
                from_obj=glob.assignment_target,
                current_module_name=current_filename[:-3],
                current_object_name=current_object_name,
            ):
                import_statement = create_relative_import_statement(
                    from_mod=module_name,
                    from_obj=glob.assignment_target,
                    current_module_name=current_filename[:-3],
                )
                glob_ctx = glob.ctx
                assert (
                    glob_ctx is not None
                ), f"No context found for {glob.assignment_target} in {current_filename}"
                is_compile_time_object = glob_ctx != TranslatedContext.VARIABLE
                # if compile time object (eg: decorator, type_spec etcn), add as module import
                if not is_compile_time_object:
                    local_import_statement_strings.append(import_statement)
                else:
                    # else add as module import
                    if (
                        not import_statement_cache.exist(
                            filename=current_filename, import_stmt=import_statement
                        )
                        and import_statement not in old_import_statements
                    ):
                        module_import_statement_strings.append(import_statement)
                        import_statement_cache.cache(
                            filename=current_filename, import_stmt=import_statement
                        )
        else:
            # if the global belongs to the current module, add it to the global_assignment_strings
            if not global_statement_cache.exist(
                filename=current_filename, glob_stmt=glob.assignment_str
            ):
                global_assignment_strings.append(glob.assignment_str)
                global_statement_cache.cache(
                    filename=current_filename, glob_stmt=glob.assignment_str
                )

        if glob.global_dependencies:
            # handle global dependencies. Dependencies are defined as:
            # GLOB = Translated_Foo(x=10, y=Translated_Bar(x=10))
            # In this case, we need to add imports for `Translated_Foo` and `Translated_Bar`
            # inside the module wherein we are injecting the global `GLOB` assignment.
            dependency_imports = []

            for from_obj_str, file in glob.global_dependencies.items():
                assert file.endswith(".py"), "filename must be a .py file"
                from_mod = file[:-3]
                if _validate_from_import(
                    from_mod=from_mod,
                    from_obj=from_obj_str,
                    current_module_name=glob.global_filename[:-3],
                    current_object_name=current_object_name,
                ):
                    import_statement = create_relative_import_statement(
                        from_mod=from_mod,
                        from_obj=from_obj_str,
                        current_module_name=glob.global_filename[:-3],
                    )
                    if not import_statement_cache.exist(
                        filename=glob.global_filename, import_stmt=import_statement
                    ):

                        dependency_imports.append(import_statement)
                        import_statement_cache.cache(
                            filename=glob.global_filename, import_stmt=import_statement
                        )

            if dependency_imports:
                if glob.global_filename == current_filename:
                    # if the global object is in the same file, we can add dependency imports
                    # to the module imports rather than manually injecing them
                    dependency_imports = [
                        imp
                        for imp in dependency_imports
                        if imp not in old_import_statements
                    ]
                    module_import_statement_strings.extend(dependency_imports)
                else:
                    # if the global object is in a different file, we need to inject the imports
                    fullpath = FileNameStrategy.create_module_structure(
                        output_dir, glob.global_filename, target
                    )
                    file_path = fullpath
                    dependency_import_strings = "\n".join(dependency_imports) + "\n\n"
                    with open(file_path, "r", encoding="utf-8", newline="\n") as file:
                        content = file.read()
                    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
                        new_source = dependency_import_strings + "\n" + content + "\n"
                        # check for any syntax errors
                        check_syntax(new_source)
                        file.seek(0, 0)
                        file.write(new_source)

    if local_import_statement_strings:
        import_nodes = _create_local_imports(local_import_statement_strings)
        if object_like.type == Types.FunctionType:
            func_name = NAME_GENERATOR.get_name(object_like)
            _inject_local_imports_in_function(import_nodes, func_name, ast_root)
        else:
            _inject_local_imports_in_class(import_nodes, ast_root)

    global_statements = (
        ""
        if not global_assignment_strings
        else "\n".join(global_assignment_strings) + "\n\n"
    )
    module_import_statements = (
        ""
        if not module_import_statement_strings
        else "\n".join(module_import_statement_strings) + "\n\n"
    )
    return global_statements, module_import_statements


def convert_absolute_to_relative_import(from_mod: str, current_module_name: str) -> str:
    """
    Convert an absolute import to a relative import based on the current module's location.

    Args:
    from_mod (str): The module from which to import (e.g., 'ivy.functional.backends.tensorflow.general')
    current_module_name (str): The name of the current module (e.g., 'ivy.functional.ivy.general')

    Returns:
    str: The relative import path
    """
    from_parts = from_mod.split(".")
    current_parts = current_module_name.split(".")

    # Find the common prefix
    common_prefix_length = 0
    for fp, cp in zip(from_parts, current_parts):
        if fp == cp:
            common_prefix_length += 1
        else:
            break

    # Calculate the number of levels to go up
    levels_up = len(current_parts) - common_prefix_length

    # Construct the relative import
    relative_prefix = "." * levels_up
    relative_path = ".".join(from_parts[common_prefix_length:])

    return f"{relative_prefix}{relative_path}"


def create_relative_import_statement(
    from_mod: str, from_obj: str, current_module_name: str, asname: Optional[str] = None
) -> str:
    """
    Create a relative import statement.

    Args:
    from_mod (str): The module from which to import
    from_obj (str): The object to import
    current_module_name (str): The name of the current module
    asname (str, optional): The alias for the imported object

    Returns:
    str: The relative import statement
    """
    relative_path = convert_absolute_to_relative_import(from_mod, current_module_name)
    if asname:
        return f"from {relative_path} import {from_obj} as {asname}"
    return f"from {relative_path} import {from_obj}"


def generate_source_code(
    ast_root,
    object_like: BaseObjectLike,
    globals: list,
    imports: set,
    from_imports: set,
    circular_reference_object_likes: List[Union["FuncObjectLike", "TypeObjectLike"]],
    source: str,
    target: str,
    object_like_bytes_to_translated_object_str_cache: ObjectLikeBytesToTranslatedObjectStringCache = None,
    import_statement_cache: ImportStatementCache = None,
    global_statement_cache: GlobalStatementCache = None,
    emitted_source_cache: EmittedSourceCache = None,
    output_dir: str = "",
    base_output_dir: str = "",
    from_cache: bool = False,
) -> str:
    """
    Main function used for generating the source code for a given function/class.
    The function processes imports, dependencies and globals and spits out source code
    corresponding to the transformed Abstract Syntax Tree (AST) as a Python module
    in the specified output directory.
    """
    original_object_like_name = NAME_GENERATOR.get_name(object_like)
    assert (
        original_object_like_name is not None
    ), f"no name associated with the object: {original_object_like_name}"
    translated_calls = get_translated_nodes(ast_root)
    filename = FileNameStrategy.infer_filename_from_object_like(
        object_like, target, base_output_dir=base_output_dir
    )
    fullpath = FileNameStrategy.create_module_structure(output_dir, filename, target)
    inject_stateful_import = True

    # check if the objectlike has already been emitted within <filename>.py.
    if emitted_source_cache.exist(
        filename=filename, obj_hash=object_like.get_object_hash()
    ):
        return original_object_like_name

    # add the objectlike to the cache as we will now proceed with emitting the source code for it.
    emitted_source_cache.cache(
        filename=filename, obj_hash=object_like.get_object_hash()
    )

    filepath = fullpath
    is_new_file = not os.path.exists(filepath)
    if is_new_file:
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            imports_to_inject = (
                _inject_standard_imports(
                    object_like,
                    target=target,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=filename,
                )
                + _inject_builtin_imports(
                    imports=imports,
                    from_imports=from_imports,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=filename,
                    from_cache=from_cache,
                )
                + _maybe_inject_stateful_import(
                    target=target,
                    inject_import=inject_stateful_import,
                    object_like=object_like,
                    ast_root=ast_root,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    filename=filename,
                )
                + _inject_module_dependencies(
                    translated_strings=translated_calls,
                    target=target,
                    object_like_bytes_to_translated_object_str_cache=object_like_bytes_to_translated_object_str_cache,
                    import_statement_cache=import_statement_cache,
                    old_imports="",
                    circular_reference_object_likes=circular_reference_object_likes,
                    object_like=object_like,
                    current_object_name=original_object_like_name,
                    current_filename=filename,
                    ast_root=ast_root,
                    base_output_dir=base_output_dir,
                )
            )

            global_statements, global_imports = (
                add_global_assignments_and_create_imports(
                    ast_root=ast_root,
                    object_like=object_like,
                    global_objects=globals,
                    global_statement_cache=global_statement_cache,
                    import_statement_cache=import_statement_cache,
                    imports=imports,
                    from_imports=from_imports,
                    old_imports="",
                    current_filename=filename,
                    current_object_name=original_object_like_name,
                    output_dir=output_dir,
                    from_cache=from_cache,
                    target=target,
                )
            )
            standard_globals = _inject_standard_globals(
                object_like,
                source=source,
                target=target,
                output_dir=output_dir,
                global_statement_cache=global_statement_cache,
                filename=filename,
            )
            code = ast_to_source_code(ast_root)
            if any(
                init_files in filename
                for init_files in (
                    "ivy.__init__",
                    "frontends.torch.__init__",
                    "frontends.numpy.__init__",
                )
            ):
                # no need to add extra imports inside ivy.__init__, ivy.functional.frontends.torch.__init__ etc.
                imports_to_inject = "\nimport ivy\n" if target == "ivy" else ""
                standard_globals = ""
            source = (
                imports_to_inject
                + global_imports
                + standard_globals
                + global_statements
                + code
            )
            # check for syntax errors
            check_syntax(source)
            f.write(source)
    else:
        # Before reading the old source, inject
        # any standard frontend globals (which are
        # directly injected by writing into the file
        # right now) so that they can be picked up
        # when reading and spliting the old source
        # rather than overwriting them with the new source
        _maybe_inject_frontend_standard_globals(
            source=source,
            target=target,
            output_dir=output_dir,
            base_output_dir=base_output_dir,
            global_statement_cache=global_statement_cache,
        )

        with open(filepath, "r", encoding="utf-8", newline="\n") as f:
            old_source = f.read()

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            old_imports, old_source_code_and_globals = _split_imports_globals_and_code(
                source=old_source
            )
            imports_to_inject = _inject_builtin_imports(
                imports=imports,
                from_imports=from_imports,
                import_statement_cache=import_statement_cache,
                old_imports=old_imports,
                filename=filename,
                from_cache=from_cache,
            ) + _inject_standard_imports(
                object_like,
                target=target,
                import_statement_cache=import_statement_cache,
                old_imports="",
                filename=filename,
            )
            global_statements, global_imports = (
                add_global_assignments_and_create_imports(
                    ast_root=ast_root,
                    object_like=object_like,
                    global_objects=globals,
                    global_statement_cache=global_statement_cache,
                    import_statement_cache=import_statement_cache,
                    imports=imports,
                    from_imports=from_imports,
                    old_imports=old_imports,
                    current_filename=filename,
                    current_object_name=original_object_like_name,
                    output_dir=output_dir,
                    from_cache=from_cache,
                    target=target,
                )
            )

            standard_globals = _inject_standard_globals(
                object_like,
                source=source,
                target=target,
                output_dir=output_dir,
                global_statement_cache=global_statement_cache,
                filename=filename,
            )
            combined_imports = _sort_statements(
                old_imports + imports_to_inject + global_imports
            )
            # TODO: remove this hardcoded check once unwanted torch code inside TF_LSTM has been removed
            if any(cls in original_object_like_name for cls in ("RNN", "LSTM")):
                combined_imports += "\nimport torch\n"
            new_dependencies = _inject_module_dependencies(
                translated_strings=translated_calls,
                target=target,
                object_like_bytes_to_translated_object_str_cache=object_like_bytes_to_translated_object_str_cache,
                import_statement_cache=import_statement_cache,
                old_imports=old_imports,
                circular_reference_object_likes=circular_reference_object_likes,
                object_like=object_like,
                current_object_name=original_object_like_name,
                current_filename=filename,
                ast_root=ast_root,
                base_output_dir=base_output_dir,
            )
            stateful_import = _maybe_inject_stateful_import(
                target=target,
                inject_import=inject_stateful_import,
                object_like=object_like,
                ast_root=ast_root,
                import_statement_cache=import_statement_cache,
                old_imports=old_imports,
                filename=filename,
            )
            source = global_statements + ast_to_source_code(ast_root)
            combined_source = old_source_code_and_globals + "\n" + source
            if any(
                init_files in filename
                for init_files in (
                    "ivy.__init__",
                    "frontends.torch.__init__",
                    "frontends.numpy.__init__",
                )
            ):
                # no need to add extra imports inside ivy.__init__, ivy.functional.frontends.torch.__init__ etc.
                combined_imports = "\nimport ivy\n" if target == "ivy" else ""
                standard_globals = ""
            new_source = (
                combined_imports
                + "\n"
                + stateful_import
                + "\n"
                + new_dependencies
                + "\n"
                + standard_globals
                + "\n"
                + combined_source
            )
            # check for syntax errors
            check_syntax(new_source)
            f.seek(0, 0)
            f.write(new_source)

    return original_object_like_name


def _split_imports_globals_and_code(source: str) -> Tuple[str, str]:
    """
    Split Python source code into imports and the rest of the code (including globals).

    This function separates the import statements from the rest of the code in a given
    Python source string. It considers only the top-level imports at the beginning of
    the file, stopping at the first non-import, non-empty line.

    Parameters:
    source (str): A string containing Python source code.

    Returns:
    tuple: A tuple containing two strings:
        - imports (str): A string of all import statements, each on a new line.
        - code_and_globals (str): A string of the remaining code, including global
          variables and function/class definitions.

    Note:
    - The function assumes that all import statements are at the top of the file.
    - Empty lines at the beginning of the file are ignored.
    - The returned strings include a trailing newline character.

    Example:
    >>> source_code = '''
    ... import numpy as np
    ... from scipy import stats
    ...
    ... GLOBAL_CONSTANT = 42
    ...
    ... def some_function():
    ...     pass
    ... '''
    >>> imports, code = _split_imports_globals_and_code(source_code)
    >>> print("Imports:")
    >>> print(imports)
    >>> print("Code and globals:")
    >>> print(code)
    Imports:
    import numpy as np
    from scipy import stats

    Code and globals:
    GLOBAL_CONSTANT = 42

    def some_function():
        pass
    """
    ast_tree = gast.parse(source)

    import_nodes = []
    non_import_nodes = []

    # Traverse the AST and classify nodes into imports and non-imports
    for node in ast_tree.body:
        if isinstance(node, (gast.Import, gast.ImportFrom)):
            import_nodes.append(node)
        else:
            non_import_nodes.append(node)

    # Unparse the nodes back into source code
    imports = [ast_to_source_code(import_node).strip() for import_node in import_nodes]
    code_and_globals = [
        ast_to_source_code(non_import_node).strip()
        for non_import_node in non_import_nodes
    ]

    return ("\n".join(imports) + "\n"), ("\n".join(code_and_globals) + "\n")
