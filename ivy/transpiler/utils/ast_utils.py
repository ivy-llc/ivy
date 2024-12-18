# global
import ast
import inspect
import types
import os
import sys
import re
import textwrap
from typing import Union, List, Dict, Set, Tuple, Optional, TYPE_CHECKING
from collections.abc import Iterable
import importlib
from types import ModuleType
from enum import Enum, auto
from packaging.version import parse

# local
import astor
import gast
from ..translations.data.object_like import BaseObjectLike
from .api_utils import (
    get_native_module_str_from_backend,
    SUPPORTED_BACKENDS_PREFIX,
    TRANSLATED_OBJ_PREFIX,
)
from .cache_utils import (
    GlobalStatementCache,
    ImportStatementCache,
    ObjectLikeBytesToTranslatedObjectStringCache,
    EmittedSourceCache,
)
from .naming_utils import NAME_GENERATOR
from .type_utils import Types


MODULE_TO_ALIAS = {
    "numpy": "np",
    "tensorflow": "tf",
}
TRANSLATED_OUTPUTS_SUBDIR = [
    "torch_frontend_outputs",
    "ivy_outputs",
    "tensorflow_outputs",
    "jax_outputs",
    "numpy_outputs",
]
FRONTEND_STANDARD_GLOBALS = {}
FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE = {
    "torch_promotion_table": "ivy.functional.frontends.torch.__init__",
    "numpy_promotion_table": "ivy.functional.frontends.numpy.__init__",
    "numpy_str_to_type_table": "ivy.functional.frontends.numpy.__init__",
    "numpy_scalar_to_dtype": "ivy.functional.frontends.numpy.__init__",
    "numpy_dtype_to_scalar": "ivy.functional.frontends.numpy.__init__",
    "numpy_casting_rules": "ivy.functional.frontends.numpy.__init__",
}
IVY_STANDARD_GLOBALS = {}
IVY_STANDARD_GLOBALS_TARGET_TO_MODULE = {
    "promotion_table": "ivy.__init__",
    "array_api_promotion_table": "ivy.__init__",
}
TF_DUNDERS_MONKEY_PATCH = textwrap.dedent(
    """
from .ivy.functional.frontends.torch.tensor import tensorflow___add___frnt_, tensorflow___sub___frnt_, tensorflow___mul___frnt_, tensorflow___truediv___frnt_, tensorflow___eq___frnt_, tensorflow___ne___frnt_
import tensorflow 

def _define_dunders(orig_method_name):
    original_method = getattr(tensorflow.Tensor, orig_method_name)
    patched_method = {
        '__add__': tensorflow___add___frnt_,
        '__sub__': tensorflow___sub___frnt_,
        '__mul__': tensorflow___mul___frnt_,
        '__truediv__': tensorflow___truediv___frnt_,
        '__eq__': tensorflow___eq___frnt_,
        '__ne__': tensorflow___ne___frnt_,
    }[orig_method_name]

    if orig_method_name in ['__eq__', '__ne__']:
        def impl(self, rhs):
            try:
                res = original_method(self, rhs)
                if isinstance(rhs, (list, tuple)):
                    return False if orig_method_name == '__eq__' else True
                return res
            except Exception:
                return patched_method(self, rhs)
    else:
        def impl(self, rhs):
            try:
                return original_method(self, rhs)
            except Exception:
                return patched_method(self, rhs)

    setattr(tensorflow.Tensor, orig_method_name, impl)

def _define_properties(orig_property_name):
    def device_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.device
        else:
            return self.device

    import keras 
    if keras.__version__ >= '3.0.0':
        patched_method = {
            'device': device_getter,
        }[orig_property_name]
        setattr(tensorflow.keras.Variable, orig_property_name, property(patched_method))

for orig_method_name in ['__add__', '__sub__', '__mul__', '__truediv__', '__eq__', '__ne__']:
    _define_dunders(orig_method_name)

for property_name in ['device']:
    _define_properties(property_name)
"""
)

JAX_DUNDER_PROPERTY_PATCH = textwrap.dedent(
    """
from .ivy.functional.frontends.torch.tensor import jax___add___frnt_, jax___sub___frnt_, jax___mul___frnt_, jax___truediv___frnt_, jax___eq___frnt_, jax___ne___frnt_
import jax 
import jaxlib 
import flax.nnx as nnx 

def _define_dunders(orig_method_name):
    original_method = getattr(jaxlib.xla_extension.ArrayImpl, orig_method_name)
    patched_method = {
        '__add__': jax___add___frnt_,
        '__sub__': jax___sub___frnt_,
        '__mul__': jax___mul___frnt_,
        '__truediv__': jax___truediv___frnt_,
        '__eq__': jax___eq___frnt_,
        '__ne__': jax___ne___frnt_,
    }[orig_method_name]

    def impl(self, rhs):
        try:
            return original_method(self, rhs)
        except Exception as e:
            return patched_method(self, rhs)

    setattr(jaxlib.xla_extension.ArrayImpl, orig_method_name, impl)

def _define_properties(orig_property_name):
    def device_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else list(self.value.devices())[0]
        else:
            return list(self.devices())[0]

    def shape_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.shape
        else:
            return original_property.__get__(self)

    def dtype_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.dtype
        else:
            return original_property.__get__(self)

    def custom_getattr(self, name):
        if name in ('shape', 'device', 'dtype', 'ndim', 'size', 'itemsize', 'T'):
            value = getattr(self, 'value')
            if value is not None:
                # Attempt to retrieve the attribute from the wrapped object (`value`)
                return getattr(value, name)
        return object.__getattribute__(self, name)
    original_property = getattr(jaxlib.xla_extension.ArrayImpl, orig_property_name, None)
    patched_method = {
        'device': device_getter,
        'shape': shape_getter,
        'dtype': dtype_getter,
    }[orig_property_name]
    
    setattr(jaxlib.xla_extension.ArrayImpl, orig_property_name, property(patched_method))
    setattr(nnx.Variable, orig_property_name, property(patched_method))
    setattr(nnx.Variable, '__getattr__', custom_getattr)

for orig_method_name in ['__add__', '__sub__', '__mul__', '__truediv__', '__eq__', '__ne__']:
    _define_dunders(orig_method_name)

for property_name in ['shape', 'dtype', 'device']:
    _define_properties(property_name)

    """
)

BACKEND_STANDARD_GLOBALS = {
    "tensorflow": [
        "\ntf.experimental.numpy.experimental_enable_numpy_behavior(True)\n",
    ],
    "jax": [],
    "numpy": [],
}

MONKEY_PATCH_GLOBALS = {
    "tensorflow": f"\n{TF_DUNDERS_MONKEY_PATCH}\n",
    "jax": f"\n{JAX_DUNDER_PROPERTY_PATCH}\n",
    "numpy": "\n",
}

if TYPE_CHECKING:
    from ivy.transpiler.translations.data.global_like import (
        GlobalObjectLike,
    )
    from ivy.transpiler.translations.data.object_like import (
        TypeObjectLike,
        FuncObjectLike,
    )


class TranslatedContext(Enum):
    VARIABLE = auto()
    DECORATOR = auto()
    BASE = auto()
    CLASS_ATTRIBUTE = auto()
    FUNCTION_ARGS = auto()
    TYPE_SPEC = auto()


class TranslatedFunctionVisitor(gast.NodeVisitor):
    """
    A visitor class that traverses the abstract syntax tree (AST) to identify
    and collect names of translated objects that the current AST references,
    along with their context.

    Attributes:
        translated_nodes (dict): A dictionary where key is the translated object names
                                 and value is the TranslatedContext enums
                                 representing the context in which it appears.
    """

    def __init__(self):
        self.translated_nodes = {}
        self.context_stack = [TranslatedContext.VARIABLE]

    def add_translated_node(self, node_id, context):
        if node_id not in self.translated_nodes:
            self.translated_nodes[node_id] = context
        elif self.translated_nodes[node_id].value < context.value:
            # if the current context takes greater precedence, then also update the context
            self.translated_nodes[node_id] = context
        else:
            # do nothing
            pass

    def push_context(self, context):
        """Push a new context onto the stack."""
        self.context_stack.append(context)

    def pop_context(self):
        """Pop the current context from the stack."""
        if self.context_stack:
            return self.context_stack.pop()

    def current_context(self):
        """Get the current context."""
        return self.context_stack[-1] if self.context_stack else None

    def visit_ClassDef(self, node):
        # Handle base classes
        self.push_context(TranslatedContext.BASE)
        for base in node.bases:
            self.visit(base)
        self.pop_context()

        # Handle keywords
        self.push_context(TranslatedContext.BASE)
        for keyword in node.keywords:
            self.visit(keyword)
        self.pop_context()

        # Handle decorators
        self.push_context(TranslatedContext.DECORATOR)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.pop_context()

        # Visit the rest of the class body
        self.push_context(TranslatedContext.CLASS_ATTRIBUTE)
        for item in node.body:
            self.visit(item)
        self.pop_context()

    def visit_FunctionDef(self, node):
        # Handle decorators
        self.push_context(TranslatedContext.DECORATOR)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.pop_context()

        # Handle function arguments and type annotations
        self.push_context(TranslatedContext.FUNCTION_ARGS)
        self.visit(node.args)
        self.pop_context()

        # Visit the return type if available
        if node.returns:
            self.push_context(TranslatedContext.TYPE_SPEC)
            self.visit(node.returns)
            self.pop_context()

        # Visit the rest of the function body
        self.push_context(TranslatedContext.VARIABLE)
        for item in node.body:
            self.visit(item)
        self.pop_context()

    def visit_arguments(self, node):
        # Visit each argument type spec (if present)
        for arg in node.args + node.kwonlyargs + node.posonlyargs:
            if arg.annotation:
                self.push_context(TranslatedContext.TYPE_SPEC)
                self.visit(arg.annotation)
                self.pop_context()

            self.push_context(TranslatedContext.FUNCTION_ARGS)
            self.visit(arg)
            self.pop_context()

        # Visit default values  (if present)
        for default in node.defaults + node.kw_defaults:
            if default:
                self.push_context(TranslatedContext.FUNCTION_ARGS)
                self.visit(default)
                self.pop_context()

        # Visit variable args and kwarg (if present)
        if node.vararg:
            if node.vararg.annotation:
                self.push_context(TranslatedContext.TYPE_SPEC)
                self.visit(node.vararg.annotation)
                self.pop_context()

            self.push_context(TranslatedContext.FUNCTION_ARGS)
            self.visit(node.vararg)
            self.pop_context()

        if node.kwarg:
            if node.kwarg.annotation:
                self.push_context(TranslatedContext.TYPE_SPEC)
                self.visit(node.kwarg.annotation)
                self.pop_context()

            self.push_context(TranslatedContext.FUNCTION_ARGS)
            self.visit(node.kwarg)
            self.pop_context()

        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # visit  annotated assignments (like class attributes with type hints)
        if node.annotation:
            self.push_context(TranslatedContext.TYPE_SPEC)
            self.visit(node.annotation)
            self.pop_context()

        if node.value:
            self.visit(node.value)

        self.visit(node.target)

    def visit_Name(self, node):
        if node.id.startswith(NAME_GENERATOR.new_prefix):
            ctx = self.current_context()
            # If we're in a class body and this is not a type specification,
            # it's a class attribute
            if ctx == TranslatedContext.CLASS_ATTRIBUTE and isinstance(
                node.ctx, gast.Store
            ):
                self.add_translated_node(node.id, TranslatedContext.CLASS_ATTRIBUTE)
            else:
                self.add_translated_node(node.id, ctx)


class VariableCaptureVisitor(gast.NodeVisitor):
    """
    A visitor class that traverses the abstract syntax tree (AST) to identify
    and collect variable names within function definitions. This includes
    positional arguments, keyword arguments, and assigned variables, while also
    tracking non-local and global variables to exclude them from the captured set.

    Attributes:
        variables (set): A set of strings representing the names of captured variables
                         within the function/class scope (not including nonlocals and globals).
        non_locals_and_globals (set): A set of strings representing the names of non-local
                                      and global variables
    """

    def __init__(self):
        self.variables = set()
        self.non_locals_and_globals = set()

    def visit_FunctionDef(self, node):
        # Add positional-only arguments
        for arg in node.args.posonlyargs:
            arg_name = ast_to_source_code(arg).strip()
            if arg_name not in self.non_locals_and_globals:
                self.variables.add(arg_name)

        # Add regular positional arguments
        for arg in node.args.args:
            arg_name = ast_to_source_code(arg).strip()
            if arg_name not in self.non_locals_and_globals:
                self.variables.add(arg_name)

        # Add keyword-only arguments
        for kwarg in node.args.kwonlyargs:
            kwarg_name = ast_to_source_code(kwarg).strip()
            if kwarg_name not in self.non_locals_and_globals:
                self.variables.add(kwarg_name)

        # Add variable positional arguments (*args)
        if node.args.vararg is not None:
            vararg_name = ast_to_source_code(node.args.vararg).strip()
            if vararg_name not in self.non_locals_and_globals:
                self.variables.add(vararg_name)

        # Add variable keyword arguments (**kwargs)
        if node.args.kwarg is not None:
            kwarg_name = ast_to_source_code(node.args.kwarg).strip()
            if kwarg_name not in self.non_locals_and_globals:
                self.variables.add(kwarg_name)

        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            self._add_target_variables(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._add_target_variables(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._add_target_variables(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._add_target_variables(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._add_target_variables(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self._add_target_variables(node.name)
        self.generic_visit(node)

    def visit_Comprehension(self, node):
        self._add_target_variables(node.target)
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        for name in node.names:
            self.non_locals_and_globals.add(name)

    def visit_Global(self, node):
        for name in node.names:
            self.non_locals_and_globals.add(name)

    def _add_target_variables(self, target):
        if isinstance(target, gast.Tuple):
            for elem in target.elts:
                elem_name = ast_to_source_code(elem).strip()
                if elem_name not in self.non_locals_and_globals:
                    self.variables.add(elem_name)
        else:
            target_name = ast_to_source_code(target).strip()
            if target_name not in self.non_locals_and_globals:
                self.variables.add(target_name)


def extract_target_object_name(name):
    """
    Extracts the target object name by removing specific prefixes and suffixes from the input name.

    Args:
        name (str): The input name from which to extract the target object name.

    Returns:
        str: The cleaned name with prefixes and suffixes removed.

    This function:
        - Removes the prefix if it matches the `new_prefix` or `old_prefix` from `NAME_GENERATOR`.
        - Removes specific suffixes like "_bknd", "_frnt", and numeric suffixes (e.g., "base_count_1", "base_count_2").
    """

    def remove_prefix(s):
        if s.startswith(NAME_GENERATOR.new_prefix):
            return s[len(NAME_GENERATOR.new_prefix) :]
        elif s.startswith(NAME_GENERATOR.old_prefix):
            return s[len(NAME_GENERATOR.old_prefix) :]
        else:
            return s

    def remove_suffix(s):
        # Remove numeric suffix(eg: _base_count_1, _base_count_2 etc.) if present
        s = re.sub(r"(_base_count_\d)+$", "", s)
        # Remove frontend/backend suffix
        pattern = r"_bknd_|_bknd|_frnt_|_frnt"
        s = re.sub(pattern, "", s)
        return s

    return remove_suffix(remove_prefix(name))


class ObjectOrderVisitor(gast.NodeVisitor):
    """
    A visitor class that traverses the AST (Abstract Syntax Tree) to collect
    and record the order of class, function, and global variable definitions.

    Attributes:
        order (dict): A dictionary where keys are tuples describing the type of the object
                      ("class", "function", or "global") and the object's name, and values
                      are the corresponding AST nodes.
    """

    def __init__(self):
        self.order = {}

    def visit_ClassDef(self, node):
        object_name = extract_target_object_name(node.name)
        self.order[("class", object_name)] = node

    def visit_FunctionDef(self, node):
        object_name = extract_target_object_name(node.name)
        self.order[("function", object_name)] = node

    def visit_For(self, node):
        key = ast_to_source_code(node.target).strip()
        self.order[("global", key)] = node

    def visit_Call(self, node):
        call_name = ast_to_source_code(node.func).strip()
        self.order[("global", call_name)] = gast.Expr(value=node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, gast.Name):
            self.order[("global", node.target.id)] = node
        elif isinstance(node.target, gast.Attribute):
            self.order[("global", node.target.attr)] = node

    def visit_Call(self, node):
        call_name = ast_to_source_code(node.func).strip()
        self.order[("global", call_name)] = gast.Expr(value=node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, gast.Name):
                self.order[("global", target.id)] = node
            elif isinstance(target, gast.Attribute):
                target_name = ast_to_source_code(target).strip()
                if any(
                    suffix in target_name
                    for suffix in [".partial_mixed_handler", ".compos"]
                ):
                    self.order[("partial_global", target_name)] = node
                else:
                    self.order[("global", target.attr)] = node

            elif isinstance(target, gast.Subscript):
                self.order[("global", ast_to_source_code(target).strip())] = node


def get_object_order(module_ast, return_as_dict=True):
    """
    Extracts and returns the order of class, function, and global variable definitions
    from the given AST of a module.

    Args:
        module_ast (gast.AST): The root of the AST representing the module.
        return_as_dict (bool): If True, returns a dictionary where the keys are tuples
                               describing the type ("class", "function", or "global") and
                               the object name, and the values are the corresponding AST nodes.
                               If False, returns a list of the object keys (type, name).

    Returns:
        Union[Dict[Tuple[str, str], gast.AST], List[Tuple[str, str]]]:
            Either a dictionary with object type and name as keys and AST nodes as values,
            or a list of the object type-name tuples.

    Example:
        >>> import gast
        >>> source_code = '''
        ... class MyClass:
        ...     def my_method(self):
        ...         pass
        ...
        ... def my_function():
        ...     pass
        ... '''
        >>> module_ast = gast.parse(source_code)
        >>> object_order = get_object_order(module_ast, return_as_dict=True)
        >>> print(object_order.keys())
        dict_keys([('class', 'MyClass'), ('function', 'my_function')])
    """
    visitor = ObjectOrderVisitor()
    visitor.visit(module_ast)
    if not return_as_dict:
        return list(visitor.order.keys())
    return visitor.order


class GlobalsVisitor(gast.NodeVisitor):
    """
    A visitor class that traverses the abstract syntax tree (AST) to identify
    and collect global variable assignments within a module. This class tracks
    global variables and associates them with their assignment expressions and
    the module they are defined in.

    Attributes:
        globals (dict): A dictionary where keys are global variable names (possibly
                        prefixed) and values are tuples containing the assignment
                        expression as a string and the module name.
        module_str (str): The name of the module being analyzed.
        prefix (str): A prefix to be added to variable names, useful for distinguishing
                      between different scopes or contexts.
    """

    def __init__(self, module, prefix=""):
        self.globals = dict()
        self.module_str = module.__name__
        self.prefix = prefix

    def visit_FunctionDef(self, node):
        pass

    def visit_ClassDef(self, node):
        pass

    def visit_Name(self, node):
        name = node.id
        if (self.prefix + name) not in self.globals:
            self.globals[self.prefix + name] = (None, None)
        return name

    def visit_Assign(self, node):
        assign_str = ast_to_source_code(node).strip()
        if ".partial_mixed_handler" in assign_str:
            return node
        for target in node.targets:
            if isinstance(target, gast.Name):  # only handle simple targets for now.
                name = self.visit(target)
                assert (
                    name is not None
                ), f"target should contain a var but found {ast_to_source_code(target)}"
                self.globals[self.prefix + name] = (assign_str, self.module_str)
        self.generic_visit(node)


def parse_source_code(module):
    try:
        source_code = inspect.getsource(module)
        return gast.parse(source_code)
    except (TypeError, OSError):
        return None


def get_module(name, package=None):
    try:
        if package:
            name = package + "." + name
        return importlib.import_module(name)
    except ImportError:
        return None


class GlobalAssignmentVisitor(gast.NodeVisitor):
    """
    A visitor class that traverses the abstract syntax tree (AST) to identify
    and collect the assignment of a specific global variable within a module.
    It searches for the target variable via:
    1) searching in the current module's global scope
    2) recursing into imported modules and searching for the variable in their global scope

    Attributes:
        target_str (str): The name of the target global variable to find.
        module (module): The module being analyzed.
        package (str): The package to which the module belongs.
        visited_modules (set): A set of modules that have already been visited to avoid cyclic dependencies.
        assignment_str (str): The assignment expression of the target variable as a string.
        module_str (str): The name of the module where the assignment was found.

    """

    def __init__(self, target_str, module, package, visited_modules):
        self.target_str = target_str
        self.module = module
        self.package = package
        self.visited_modules = visited_modules
        self.assignments = []
        self.module_str = None

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == self.target_str or alias.asname == self.target_str:
                parts = alias.name.split(".")
                if len(parts) > 1:
                    module_name = ".".join(parts[:-1])
                    object_name = parts[-1]
                    if module_name not in self.visited_modules:
                        module = get_module(module_name, self.package)
                        if module and not inspect.ismodule(
                            getattr(module, self.target_str, None)
                        ):
                            self.visited_modules.add(module_name)
                            result = get_global_assignment(
                                module, object_name, self.visited_modules
                            )
                            if result != (None, None):
                                if result[0] not in self.assignments:
                                    self.assignments.extend(result[0])
                                    self.module_str = result[1]
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.names[0].name == "*":
            module_name = node.module
            package_name = (
                self.package
                if ast_to_source_code(node).split(" ")[1].startswith(".")
                else None
            )
            if module_name not in self.visited_modules:
                module = get_module(module_name, package_name)
                if module and not inspect.ismodule(
                    getattr(module, self.target_str, None)
                ):
                    if self.target_str in dir(module):
                        self.visited_modules.add(module_name)
                        result = get_global_assignment(
                            module, self.target_str, self.visited_modules
                        )
                        if result != (None, None):
                            if result[0] not in self.assignments:
                                self.assignments.extend(result[0])
                                self.module_str = result[1]
                        if self.assignments:
                            return
        else:
            for alias in node.names:
                if alias.name == self.target_str or alias.asname == self.target_str:
                    target_str = alias.name
                    if not node.module:
                        module_name = self.package
                        package_name = None
                    else:
                        module_name = node.module
                        from_module_str = ast_to_source_code(node).split(" ")[1]
                        level = from_module_str.count(".") - module_name.count(".")
                        package_name = (
                            ".".join(self.package.split(".")[: -(level - 1)])
                            if level > 1
                            else self.package if level == 1 else None
                        )
                    if module_name not in self.visited_modules:
                        module = get_module(module_name, package_name)
                        if module and not inspect.ismodule(
                            getattr(module, self.target_str, None)
                        ):
                            self.visited_modules.add(module_name)
                            result = get_global_assignment(
                                module, target_str, self.visited_modules
                            )
                            if result != (None, None):
                                if result[0] not in self.assignments:
                                    self.assignments.extend(result[0])
                                    self.module_str = result[1]

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        pass

    def visit_ClassDef(self, node):
        pass

    def visit_AnnAssign(self, node):
        if isinstance(node.target, gast.Name) and node.target.id == self.target_str:
            assign_str = ast_to_source_code(node).strip()
            if assign_str not in self.assignments:
                self.assignments.append(assign_str)
                self.module_str = self.module.__name__
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, gast.Name) and target.id == self.target_str:
                assign_str = ast_to_source_code(node).strip()
                if assign_str not in self.assignments:
                    self.assignments.append(assign_str)
                    self.module_str = self.module.__name__
                    break
            elif isinstance(target, gast.Subscript):
                if (
                    isinstance(target.value, gast.Name)
                    and target.value.id == self.target_str
                ):
                    assign_str = ast_to_source_code(node).strip()
                    if assign_str not in self.assignments:
                        self.assignments.append(assign_str)
                        self.module_str = self.module.__name__
                        break
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, gast.Name) and node.target.id == self.target_str:
            assign_str = ast_to_source_code(node).strip()
            if assign_str not in self.assignments:
                self.assignments.append(assign_str)
                self.module_str = self.module.__name__
        self.generic_visit(node)


class ImportObj:
    """
    Represents an imported module.

    Attributes:
        module (str): The name of the imported module.
        asname (str, optional): The alias used for the import, if any.
        canonical_name (str): The full canonical name of the import.
    """

    def __init__(self, *, module, asname=None):
        self.module = module
        self.asname = asname
        self.canonical_name = f"{self.module}"

    def __repr__(self):
        return self.canonical_name


class FromImportObj:
    """
    Represents an object imported from a module.

    Attributes:
        module (str): The name of the module from which the object is imported.
        obj (str): The name of the imported object.
        asname (str, optional): The alias used for the import, if any.
        canonical_name (str): The full canonical name of the imported object.
    """

    def __init__(self, *, module, obj, asname=None):
        self.module = module
        self.obj = obj
        self.asname = asname
        self.canonical_name = f"{self.module}.{self.obj}"

    def __repr__(self):
        return self.canonical_name


class InternalObj:
    """
    Represents an object defined within the root module.

    Attributes:
        name (str): The name of the object.
        canonical_name (str): The full canonical name of the object.
        type (str): The type of the object ('function' or 'class').
    """

    def __init__(self, *, name, module, canonical_name, type):
        self.name = name
        self.module = module
        self.canonical_name = canonical_name
        self.type = type

    def __repr__(self):
        return f"{self.type}: {self.canonical_name}"


class ImportVisitor(gast.NodeVisitor):
    """
    AST visitor that captures all imports in a given module and internal definitions.

    This visitor handles regular imports, from-imports (including relative imports),
    and internal function/class definitions.

    Attributes:
        import_dict (dict): A dictionary of regular imports.
        from_import_dict (dict): A dictionary of from-imports.
        internal_dict (dict): A dictionary of internally defined functions and classes.
        root_module (types.ModuleType): The root module being visited.
    """

    def __init__(self, module):
        self.import_dict = {}
        self.from_import_dict = {}
        self.internal_dict = {}
        self.root_module = module

    def visit_Import(self, node):
        """
        Process regular import statements.

        Args:
            node (gast.Import): The Import node to process.
        """
        for alias in node.names:
            if alias.asname:
                self.import_dict[alias.asname] = ImportObj(
                    module=alias.name, asname=alias.asname
                )
            else:
                self.import_dict[alias.name] = ImportObj(module=alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Process from-import statements, including relative imports.

        Args:
            node (gast.ImportFrom): The ImportFrom node to process.
        """
        module = self._resolve_relative_import(node)
        if module and not any(module.startswith(s) for s in TRANSLATED_OBJ_PREFIX):
            for alias in node.names:
                if alias.asname:
                    self.from_import_dict[alias.asname] = FromImportObj(
                        module=module, obj=alias.name, asname=alias.asname
                    )
                else:
                    self.from_import_dict[alias.name] = FromImportObj(
                        module=module, obj=alias.name
                    )
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """
        Process function definitions within the root module.

        Args:
            node (gast.FunctionDef): The FunctionDef node to process.
        """
        canonical_name = f"{self.root_module.__name__}.{node.name}"
        self.internal_dict[node.name] = InternalObj(
            name=node.name,
            module=self.root_module.__name__,
            canonical_name=canonical_name,
            type="function",
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """
        Process class definitions within the root module.

        Args:
            node (gast.ClassDef): The ClassDef node to process.
        """
        canonical_name = f"{self.root_module.__name__}.{node.name}"
        self.internal_dict[node.name] = InternalObj(
            name=node.name,
            module=self.root_module.__name__,
            canonical_name=canonical_name,
            type="class",
        )
        self.generic_visit(node)

    def _resolve_relative_import(self, node):
        """
        Resolve the full module path for relative imports.

        Args:
            node (gast.ImportFrom): The ImportFrom node to resolve.

        Returns:
            str: The full module path of the import.
        """
        if node.level == 0:
            return node.module

        # Remove as many parts as the relative import level
        parts = self.root_module.__name__.split(".")
        parts = parts[: -node.level]

        # If there's a module specified in the import, add it
        if node.module:
            parts.append(node.module)

        return ".".join(parts)

    def process_module(self, tree):
        """
        Process the entire module, including AST analysis and inspection.

        This method combines AST visiting with inspection to catch all defined
        objects in the module.
        """
        # First, visit the AST
        self.visit(tree)

        # Then, use inspection to catch any objects that might have been missed
        for name, obj in inspect.getmembers(self.root_module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if (
                    obj.__module__ == self.root_module.__name__
                    and name not in self.internal_dict
                ):
                    canonical_name = f"{self.root_module.__name__}.{name}"
                    obj_type = "function" if inspect.isfunction(obj) else "class"
                    self.internal_dict[name] = InternalObj(
                        name=name,
                        module=self.root_module.__name__,
                        canonical_name=canonical_name,
                        type=obj_type,
                    )


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if isinstance(ast_node, str):
        return ast_node

    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s."
            % type(ast_node)
        )
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return "".join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code


def keyword_in_keyword_args(node, keyword: str):
    return any(kw.arg == keyword for kw in node.keywords)


def is_super_call_node(node):
    """
    Check if a node represents a call to super.

    """
    _node = node if not isinstance(node, ast.AST) else gast.ast_to_gast(node)
    return (
        isinstance(_node, gast.Attribute)
        and isinstance(_node.value, gast.Call)
        and isinstance(_node.value.func, gast.Name)
        and _node.value.func.id == "super"
    )


def get_attribute_full_name(node):
    assert isinstance(
        node, gast.Attribute
    ), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def property_to_func(orig_obj, node):
    """
    Retrieves the function (getter, setter, or deleter) associated with a Python property
    based on the context of use.
    """
    # Determine the function associated with the property based on the context
    if isinstance(node.ctx, gast.Load):
        property_func = orig_obj.fget
    elif isinstance(node.ctx, gast.Store):
        property_func = orig_obj.fset
    else:
        property_func = orig_obj.fdel

    return property_func


def is_builtin_type_call_node(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_builtin_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        func = func_str.strip().split(".")[-1]
        return any([func in dir(cls) for cls in (list, dict, set, str)])
    except Exception:
        return False


def is_unpacking_assignment(node):
    if not isinstance(node, gast.Assign):
        return False

    if len(node.targets) != 1 or not isinstance(node.targets[0], gast.Tuple):
        return False

    # Check if any element in the tuple is a Name (variable)
    if any(isinstance(elt, gast.Name) for elt in node.targets[0].elts):
        # Check if the value being assigned is a Name (variable)
        if isinstance(node.value, gast.Name):
            return True

        # Check if the value being assigned is an attribute (e.g., object.attribute)
        elif isinstance(node.value, gast.Attribute):
            return True

        # Check if the value being assigned is a Subscript (e.g., list[index])
        elif isinstance(node.value, gast.Subscript):
            return True

    return False


def set_parents(node, parent=None):
    """Assign parents for each node in a gast tree."""
    node.parent = parent
    for child in gast.iter_child_nodes(node):
        set_parents(child, node)


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


def get_global_assignment(
    module: ModuleType, target_str: str, visited_modules: Set[ModuleType] = None
):
    """
    Analyzes the source code of a given module to find the assignment of a specific
    global variable. It uses the GlobalAssignmentVisitor to traverse the AST and
    identify the assignment.

    Args:
        module (module): The module to analyze.
        target_str (str): The name of the target global variable to find.
        visited_modules (set, optional): A set of modules that have already been visited
                                         to avoid cyclic dependencies. Defaults to None.

    Returns:
        tuple: A tuple containing:
               - The assignment expression of the target variable as a string.
               - The name of the module where the assignment was found.
               If the assignment is not found, returns (None, None).

    Example:
        >>> import types
        >>> module = types.ModuleType("example_module")
        >>> module.__name__ = "example_module"
        >>> module.__package__ = ""
        >>> source_code = '''
        ... my_var = 42
        ... '''
        >>> def parse_source_code(mod):
        ...     return gast.parse(source_code)
        >>> def get_module(name, package):
        ...     return module
        >>> assignment, mod_name = get_global_assignment(module, 'my_var')
        >>> print(assignment)
        "my_var = 42"
        >>> print(mod_name)
        "example_module"
    """
    if visited_modules is None:
        visited_modules = set()
    tree = parse_source_code(module)
    if tree is not None:
        visitor = GlobalAssignmentVisitor(
            target_str, module, module.__package__, visited_modules
        )
        visitor.visit(tree)
        if visitor.assignments:
            return visitor.assignments, visitor.module_str
    return None, None


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


def split_imports_globals_and_code(source: str) -> Tuple[str, str]:
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
    >>> imports, code = split_imports_globals_and_code(source_code)
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


def _inject_local_imports_in_function(
    import_nodes: List[gast.ImportFrom], func_name: str, ast_root: gast.AST
):
    """
    Inject import nodes into the Abstract Syntax Tree (AST) of a Python function.
    """

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


def _maybe_populate_frontend_standard_globals(source: str, target: str) -> None:
    if (
        target in SUPPORTED_BACKENDS_PREFIX
        or source == "torch_frontend"
        and target == "ivy"
    ):
        if target in SUPPORTED_BACKENDS_PREFIX:
            # ivy standard globals
            import ivy

            global IVY_STANDARD_GLOBALS
            IVY_STANDARD_GLOBALS["promotion_table"] = repr(ivy.promotion_table)
            IVY_STANDARD_GLOBALS["array_api_promotion_table"] = repr(
                ivy.array_api_promotion_table
            )

    # frontend standard globals
    import ivy.functional.frontends.torch
    import ivy.functional.frontends.numpy

    global FRONTEND_STANDARD_GLOBALS
    FRONTEND_STANDARD_GLOBALS["torch_promotion_table"] = repr(
        ivy.functional.frontends.torch.torch_promotion_table
    )
    FRONTEND_STANDARD_GLOBALS["numpy_promotion_table"] = repr(
        ivy.functional.frontends.numpy.numpy_promotion_table
    )
    FRONTEND_STANDARD_GLOBALS["numpy_str_to_type_table"] = repr(
        ivy.functional.frontends.numpy.numpy_str_to_type_table
    )
    # TODO: Add support translating these globals which contain custom objects from the numpy frontend
    # FRONTEND_STANDARD_GLOBALS["numpy_scalar_to_dtype"] = repr(ivy.functional.frontends.numpy.numpy_scalar_to_dtype)
    # FRONTEND_STANDARD_GLOBALS["numpy_dtype_to_scalar"] = repr(ivy.functional.frontends.numpy.numpy_dtype_to_scalar)
    FRONTEND_STANDARD_GLOBALS["numpy_casting_rules"] = repr(
        ivy.functional.frontends.numpy.numpy_casting_rules
    )


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

            global IVY_STANDARD_GLOBALS
            for target_str, assign_str in IVY_STANDARD_GLOBALS.items():
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

        global FRONTEND_STANDARD_GLOBALS
        for target_str, assign_str in FRONTEND_STANDARD_GLOBALS.items():
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


def check_syntax(source_code: str) -> bool:
    """
    Check if the provided Python source code is syntactically correct.

    Args:
        source_code (str): The source code to check.

    Returns:
        bool: True if the code is syntactically correct, False otherwise.
    """
    try:
        # Attempt to compile the source code to check for syntax errors
        compile(source_code, "<string>", "exec")
        return True
    except SyntaxError as e:
        raise SyntaxError(f"Syntax Error: {e}")


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
                        old_source = _delete_licence(old_source)

                    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
                        old_imports_str, old_source_code_and_globals = (
                            split_imports_globals_and_code(source=old_source)
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


def replace_placeholders(
    input_node: gast.AST, variable_nodes: List[gast.AST], placeholder="_"
) -> gast.AST:
    """
    Use this function to replace a recurring `placeholder`
    in an `input_node` stringified representation with all the
    stringified representations from the list of `variable_nodes` provided.
    """
    input_string = ast_to_source_code(input_node).strip()
    variable_names = [ast_to_source_code(node).strip() for node in variable_nodes]

    var_index = 0

    # Convert the string to a list for easy modification
    string_list = list(input_string)

    for i, char in enumerate(input_string):
        if char == placeholder:
            if var_index < len(variable_names):
                # Replace '_' with the next variable name
                string_list[i] = variable_names[var_index]
                var_index += 1
            else:
                # If the list of variable names is exhausted, break the loop
                break

    # Join the list back into a string
    result_string = "".join(string_list).replace("_", "'_'")
    return gast.parse(result_string).body[0].value


def reorder_objects(source_module, target_module, logger):
    """
    Reorders objects in the target module to match the order of objects in the source module.

    Args:
        source_module (str): Source module's code as a string to derive object order.
        target_module (str): Target module's code as a string to be reordered.
        logger (Logger): Logger instance used to log any mismatches between the source and target modules.

    Returns:
        str: The reordered target module code as a string.

    The function:
        - Parses the source and target modules into ASTs.
        - Extracts the order of objects (classes, functions, etc.) from the source module.
        - Reorders the objects in the target module to match the source module.
        - Logs any objects in the target module that don't match the source.
        - Returns the reordered target module as source code.
    """
    assert isinstance(source_module, str), "source_module must be a string"
    assert isinstance(target_module, str), "target_module must be a string"

    source_ast = gast.parse(source_module)
    target_ast = gast.parse(target_module)

    # Get the order of objects from the source module
    source_order = get_object_order(source_ast, return_as_dict=False)
    target_objects = get_object_order(target_ast)

    experimental_node = None
    for key, value in target_objects.items():
        if "tf.experimental.numpy.experimental_enable_numpy_behavior" in key:
            experimental_node = key
            break
    # Separate imports from the rest of the code
    imports = [
        node
        for node in target_ast.body
        if isinstance(node, (gast.Import, gast.ImportFrom))
    ]

    # Reorder objects based on source_order
    reordered_body = imports[:]
    if experimental_node:
        reordered_body.append(target_objects.pop(experimental_node))
    for obj_type, obj_name in source_order:
        key = (obj_type, obj_name)
        if key in target_objects:
            reordered_body.append(target_objects.pop(key))

    if target_objects:
        # If there are any remaining objects in target_objects, add them to the end
        key_mismatches = list(target_objects.keys())
        logger.warn(f"Not all objects were reordered: {key_mismatches}")
        reordered_body.extend(target_objects.values())

    # Create a new AST with reordered body
    new_target_ast = gast.Module(body=reordered_body, type_ignores=[])

    # Convert the AST back to source code
    new_target_code = ast_to_source_code(new_target_ast)

    return new_target_code


def reorder_module_objects(source_file_path, target_file_path, logger):

    assert isinstance(source_file_path, str) and source_file_path.endswith(
        ".py"
    ), "source_file_path must be a string ending with '.py'"
    assert isinstance(target_file_path, str) and target_file_path.endswith(
        ".py"
    ), "target_file_path must be a string ending with '.py'"

    with open(source_file_path, "r", encoding="utf-8", newline="\n") as source_file:
        source_code = source_file.read()

    with open(target_file_path, "r", encoding="utf-8", newline="\n") as target_file:
        target_code = target_file.read()

    reordered_code = reorder_objects(source_code, target_code, logger)

    with open(target_file_path, "w", encoding="utf-8", newline="\n") as target_file:
        target_file.write(reordered_code)


def _delete_licence(source_code):
    """delete the licence string from the module's source_code"""
    tree = gast.parse(source_code)
    if tree.body:
        for node in gast.walk(tree):
            if isinstance(node, gast.Expr) and isinstance(node.value, gast.Constant):
                if isinstance(node.value.value, str):
                    if node.value.value.rstrip().strip().startswith("Copyright"):
                        tree.body.remove(node)
                        break
    return ast_to_source_code(tree)


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
            old_source = _delete_licence(old_source)

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            old_imports, old_source_code_and_globals = split_imports_globals_and_code(
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
