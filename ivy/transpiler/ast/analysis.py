import gast
import inspect
import types
from typing import Dict, Set, Tuple
from collections.abc import Iterable

from .globals import TranslatedContext
from .nodes import FromImportObj, ImportObj, InternalObj
from .visitors import (
    TranslatedFunctionVisitor,
    VariableCaptureVisitor,
    GlobalsVisitor,
    ImportVisitor,
)


def get_translated_nodes(node: gast.AST) -> Dict[str, TranslatedContext]:
    """
    Analyzes the given AST node to find all translated object references. A translated
    object is one whose name starts with the prefix defined by `NAME_GENERATOR.new_prefix`.

    Args:
        node (gast.AST): The root of the AST to analyze.

    Returns:
        set: A set of strings representing the names of translated objects found in the AST.

    Example:
        >>> import gast
        >>> source_code = '''
        ... def my_function():
        ...     x = Translated_foo()
        ...     y = Translated_bar
        ... '''
        >>> node = gast.parse(source_code)
        >>> translated_nodes = get_translated_nodes(node)
        >>> print(translated_nodes)
        {'Translated_foo', 'Translated_bar'}
    """
    visitor = TranslatedFunctionVisitor()
    visitor.visit(node)
    return visitor.translated_nodes


def get_function_vars(node: gast.AST) -> Tuple[Set[str], Set[str]]:
    """
    Analyzes the given AST node to find all variable names within function definitions.
    This includes positional arguments, keyword arguments, and assigned variables,
    while also tracking non-local and global variables to exclude them from the captured set.

    Args:
        node (gast.AST): The root of the AST to analyze.

    Returns:
        tuple: A tuple containing two sets:
               - The first set contains strings representing the names of captured variables within the function definitions.
               - The second set contains strings representing the names of non-local and global variables.

    Example:
        >>> import gast
        >>> source_code = '''
        ... def my_function(a, b, *args, **kwargs):
        ...     c = 10
        ...     d, e = 20, 30
        ...     nonlocal f
        ...     global g
        ... '''
        >>> node = gast.parse(source_code)
        >>> variables, non_locals_and_globals = get_function_vars(node)
        >>> print(variables)
        {'a', 'b', 'c', 'd', 'e', 'args', 'kwargs'}
        >>> print(non_locals_and_globals)
        {'f', 'g'}
    """
    visitor = VariableCaptureVisitor()
    visitor.visit(node)
    return visitor.variables, visitor.non_locals_and_globals


def get_module_globals(modules: Iterable, prefix: str = ""):
    """
    Analyzes the source code of a given set of modules to find all global variables and their assignments.
    It uses the GlobalsVisitor to traverse the AST of each module and collects the global variables.

    Args:
        modules (Iterable): An iterable of modules or objects from which to extract the modules.
        prefix (str, optional): A prefix to prepend to the global variable names. Defaults to "".

    Returns:
        dict: A dictionary where the keys are global variable names (possibly prefixed) and the values
              are tuples containing the assignment expressions and the module names.

    Example:
        >>> import types
        >>> module = types.ModuleType("example_module")
        >>> module.__name__ = "example_module"
        >>> source_code = '''
        ... x = 10
        ... y = 20
        ... '''
        >>> def mock_getsource(mod):
        ...     return source_code
        >>> inspect.getsource = mock_getsource
        >>> def mock_parse(source_code):
        ...     return gast.parse(source_code)
        >>> gast.parse = mock_parse
        >>> globals_dict = get_module_globals([module], prefix="ivy_")
        >>> print(globals_dict)
        {'ivy_x': ('x = 10', 'example_module'), 'ivy_y': ('y = 20', 'example_module')}
    """
    assert isinstance(
        modules, Iterable
    ), f"modules must be an iterable but is of type {type(modules)}"
    all_globals = dict()
    for module in modules:
        if isinstance(module, (type, types.FunctionType)):
            module = inspect.getmodule(module)
        try:
            source_code = inspect.getsource(module)
        except (TypeError, OSError):
            continue
        tree = gast.parse(source_code)
        visitor = GlobalsVisitor(module, prefix=prefix)
        visitor.visit(tree)
        new_globals = {k: v for k, v in visitor.globals.items() if v != (None, None)}
        all_globals.update(new_globals)
    return all_globals


def get_import_dict(
    modules: Iterable,
) -> Tuple[Dict[str, ImportObj], Dict[str, FromImportObj], Dict[str, InternalObj]]:
    """
    Analyze a collection of modules and gather information about their imports and internal definitions.

    This function uses the ImportVisitor to processes each module in the given iterable, extracting information about
    regular imports, from-imports, and internally defined objects (functions and classes).

    Args:
        modules (Iterable): An iterable of modules, classes, or functions to analyze.
            If a class or function is provided, its containing module will be analyzed.

    Returns:
        Tuple[Dict[str, ImportObj], Dict[str, FromImportObj], Dict[str, InternalObj]]:
            A tuple containing three dictionaries:
            - imports: Regular imports (module: ImportObj)
            - from_imports: From-imports (name: FromImportObj)
            - internal_objects: Internally defined objects (name: InternalObj)

    Raises:
        AssertionError: If the 'modules' argument is not an iterable.
        TypeError: If unable to get the source code of a module.
        OSError: If unable to read the source file of a module.

    Example:
        >>> import types
        >>> module = types.ModuleType("example_module")
        >>> module.__name__ = "example_module"
        >>> source_code = '''
        ... import os
        ... from sys import path
        ... def func():
        ...     pass
        ... class MyClass:
        ...     pass
        ... '''
        >>> def mock_getsource(mod):
        ...     return source_code
        >>> inspect.getsource = mock_getsource
        >>> def mock_parse(source_code):
        ...     return gast.parse(source_code)
        >>> gast.parse = mock_parse
        >>> imports, from_imports, internal_objects = get_import_dict([module])
        >>> print(imports)
        {'os': ImportObj(module='os', alias=None)}
        >>> print(from_imports)
        {'path': FromImportObj(module='sys', name='path', alias=None)}
        >>> print(internal_objects)
        {'func': InternalObj(name='func', type='function'),
         'MyClass': InternalObj(name='MyClass', type='class')}
    """
    assert isinstance(
        modules, Iterable
    ), f"modules must be an iterable but is of type {type(modules)}"
    imports, from_imports, internal_objects = dict(), dict(), dict()
    for module in modules:
        if isinstance(module, (type, types.FunctionType)):
            module = inspect.getmodule(module)
        try:
            source_code = inspect.getsource(module)
        except (TypeError, OSError):
            continue
        tree = gast.parse(source_code)
        visitor = ImportVisitor(module)
        visitor.process_module(tree)
        imports.update(visitor.import_dict)
        from_imports.update(visitor.from_import_dict)
        internal_objects.update(visitor.internal_dict)
    return imports, from_imports, internal_objects
