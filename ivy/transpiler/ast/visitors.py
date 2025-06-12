import gast
import inspect
from types import ModuleType
from typing import Set

from ivy.transpiler.utils.ast_utils import (
    ast_to_source_code,
    extract_target_object_name,
    get_module,
    parse_source_code,
)
from ivy.transpiler.utils.api_utils import TRANSLATED_OBJ_PREFIX
from ivy.transpiler.utils.naming_utils import NAME_GENERATOR

from .globals import TranslatedContext
from .nodes import FromImportObj, ImportObj, InternalObj


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
