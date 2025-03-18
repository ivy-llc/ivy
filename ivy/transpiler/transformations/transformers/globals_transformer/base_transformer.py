# global
from copy import copy, deepcopy
import gast
import types
from typing import List, Tuple, Optional, Union, TYPE_CHECKING

# local
from ..base_transformer import (
    BaseTransformer,
)
from ...transformer import Transformer
from ....translations.data.global_like import (
    GlobalObjectLike,
    Position,
    StackObjectLike,
)
from ....translations.data.object_like import BaseObjectLike
from ....utils.ast_utils import (
    FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE,
    IVY_STANDARD_GLOBALS_TARGET_TO_MODULE,
    ast_to_source_code,
    get_module_globals,
    get_global_assignment,
    get_function_vars,
    get_module,
    create_relative_import_statement,
    FileNameStrategy,
    TranslatedContext,
)
from ....utils.api_utils import (
    get_function_from_modules,
    SUPPORTED_BACKENDS_PREFIX,
)
from ....utils.naming_utils import NAME_GENERATOR
from ....utils.origin_utils import ORIGI_INFO
from ....utils.conversion_utils import (
    BUILTIN_LIKELY_MODULE_NAMES,
)
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)

if TYPE_CHECKING:
    from ....translations.data.object_like import (
        TypeObjectLike,
        FuncObjectLike,
    )

# this is suboptimal. However, once a backend is set in ivy, it becomes practically
# impossible to retrieve the source code locations for ivy globals as the module now
# points to the new backend, rather than `ivy` itself. Hence, we need to populate a
# globals dict ahead-of-time in order to circumvent this issue.
import ivy

if ivy.backend_stack:
    current_backend = ivy.current_backend_str()
    ivy.unset_backend()
else:
    current_backend = None
IVY_GLOBS = get_module_globals(
    [
        ivy,
        ivy.functional.ivy.constants,
        ivy.functional.ivy.creation,
        ivy.utils.backend.handler,
    ]
)
if current_backend:
    # reset the backend to the original one
    ivy.set_backend(current_backend)

NAME_GLOBS_TO_IGNORE = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "char",
    "short",
    "int",
    "float",
    "long",
    "half",
    "bool",
    "double",
    "complex64",
    "complex128",
    "native_int8",
    "native_int16",
    "native_int32",
    "native_int64",
    "native_uint8",
    "native_uint16",
    "native_uint32",
    "native_uint64",
    "native_bfloat16",
    "native_float16",
    "native_float32",
    "native_float64",
    "native_double",
    "native_complex64",
    "native_complex128",
    "native_bool",
    "promotion_table",
    "array_api_promotion_table",
    "common_extra_promotion_table",
    "precise_extra_promotion_table",
    "array",
    "NativeArray",
    "JaxArray",
    "NativeShape",
    "NativeDtype",
    "NativeDevice",
    "NativeModule",
)
ATTR_GLOBS_TO_IGNORE = ("torch.Tensor", "ivy.Variable")


def are_modules_distinct(module1, module2):
    """
    Check if two module paths are entirely distinct.

    This function ensures that neither module is a subset of the other.

    Args:
        module1 (str): The first module path.
        module2 (str): The second module path.

    Returns:
        bool: True if the modules are distinct, False otherwise.
    """
    parts1 = module1.split(".")
    parts2 = module2.split(".")

    # Check if one is a subset of the other
    for part1, part2 in zip(parts1, parts2):
        if part1 == part2:
            return False

    return True


class BaseGlobalsTransformer(BaseTransformer):
    glob_stack = []

    """
    A class to capture globals defined and/or imported in the source code.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        # filters for modules for which we dont capture the globals. This is useful to avoid
        # capturing problematic globals which are defined within these modules
        # eg: alias globals(eg: `conv2d = torch._C.conv2d` inside torch.nn.functional)
        # TODO: move this to a common config that both Recursive Transformer and Globals Transformer
        # can use.
        self.torch_unsupported_mods = [
            "torch._C",
            "torch.fx",
            "torch.jit",
            "torch.onnx",
            "torch.autograd",
            "torch.xpu",
            "torch.hub",
            "torch.backends",
            "torch.random",
            "torch.amp",
            "torch.overrides",
            "torch.distributed",
            "torch._utils_internal",
            "torch._dynamo",
            "torch.serialization",
            "torch.nn.functional",
            "torch.functional",
        ]  # populate any frontend-specific modules(eg: TF_UNSUPPORTED_MODS, JAX_UNSUPPORTED_MODS etc..)
        # when we expand to adding new frontends to the translator
        source_unsupported_mods = (
            self.torch_unsupported_mods
        )  # source-specific modules. We dont want to capture globals defined in these modules
        target_unsupported_mods = [
            "tensorflow",
            "jax",
            "jaxlib",
            "numpy",
        ]  # target-specific modules. We dont want to capture globals defined in these modules
        self.unsupported_mods = source_unsupported_mods + target_unsupported_mods

        self.context_stack = [TranslatedContext.VARIABLE]  # default context is VARIABLE

    def transform(self):
        self.variables, self.non_locals_and_globals = get_function_vars(self.root)
        self.visit(self.root)

    def visit_Import(self, node):
        self.generic_visit(node)
        for alias in node.names:
            try:
                module = get_module(alias.name)
                if module:
                    self.transformer.object_module = module
            except ImportError:
                continue
        return node

    def visit_ImportFrom(self, node):
        self.generic_visit(node)
        try:
            from_module_str = ast_to_source_code(node).split(" ")[1]
            module_name = node.module
            filename = self.transformer.object_like.get_object_module(
                source=self.transformer.source
            ).__name__
            level = from_module_str.count(".") - module_name.count(".")
            package_name = (
                ".".join(filename.split(".")[:-level]) if level != 0 else None
            )
            module = get_module(module_name, package=package_name)
            if module:
                self.transformer.object_module = module
        except ImportError:
            pass
        return node

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

        # Handle bases
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

        return node

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

        if self.transformer.object_like.is_mixed_function:
            self._handle_mixed_functions(self.transformer.object_like)
        return node

    def visit_arguments(self, node):
        # Visit each argument type spec (if present)
        for arg in node.args + node.kwonlyargs + node.posonlyargs:
            if arg.annotation:
                self.push_context(TranslatedContext.TYPE_SPEC)
                self.visit(arg.annotation)
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
        return node

    def visit_AnnAssign(self, node):
        # visit  annotated assignments (like class attributes with type hints)
        if node.annotation:
            self.push_context(TranslatedContext.TYPE_SPEC)
            self.visit(node.annotation)
            self.pop_context()

        if node.value:
            self.visit(node.value)

        self.visit(node.target)
        return node

    def visit_Attribute(self, node):
        attr_str = ast_to_source_code(node).strip()
        if attr_str in ATTR_GLOBS_TO_IGNORE:
            return node
        self.generic_visit(node)
        *module_str, target_str = attr_str.split(".")
        module_name = ".".join(module_str)
        module_to_search = get_function_from_modules(
            module_name, self.transformer.object_module
        )
        object_to_search = get_function_from_modules(
            attr_str, self.transformer.object_module
        )
        if isinstance(module_to_search, types.ModuleType) and not isinstance(
            object_to_search, types.ModuleType
        ):
            node = self._handle_globals(node, target_str, attr_str, module_to_search)
        return node

    def visit_Name(self, node):
        name_str = ast_to_source_code(node).strip()
        target_str = name_str
        module_to_search = self.transformer.object_module
        node = self._handle_globals(node, target_str, target_str, module_to_search)
        return node

    def postprocess_origin_info_for_globals(
        self, global_obj_to_translate, new_name: str
    ):
        """
        attaches the current global object as a dependency to it's parent.
        NOTE: this version is specific for handling dependency cases when the RHS object is a
        global itself. For callable objects(ie: functions/class), see `postprocess_origin_info`
        in the BaseRecurser.


        Eg: if we have the following code:
        FullPadType = torch.Tensor # inside module A
        PadType = Union[List[Tensor], FullPadType] # inside module B

        when we translate the Tensor object, we want to attach the FullPadType object as a dependency to the PadType object (ie: its parent).
        This will allow us to correctly add necessary imports to the translated code.

        Eg: In the translated code, we will have the following code:
        # module_B.py
        from module_A import FullPadType
        PadType = Union[List[Tensor], FullPadType]
        """
        parent_node = (
            self.root.body[0] if isinstance(self.root, gast.Module) else self.root
        )
        origin_info = getattr(parent_node, ORIGI_INFO, None)
        if origin_info and origin_info.from_global:
            origin_info.global_dependencies[new_name] = (
                global_obj_to_translate.global_filename
            )
            setattr(parent_node, ORIGI_INFO, origin_info)

    def get_global_filename(
        self,
        object_like: Union["FuncObjectLike", "TypeObjectLike"],
        module: types.ModuleType,
        from_mixed_object_like: bool = False,
    ):
        """
        Get the filename wherein the current global being captures exists.
        """
        if from_mixed_object_like:
            return object_like.filename

        assert isinstance(
            module, types.ModuleType
        ), f"module must be a ModuleType. got {module}"
        module_name = FileNameStrategy.infer_filename_from_module_name(
            module.__name__,
            as_module=True,
            base_output_dir=object_like.base_output_dir,
        )
        if (
            not module_name.endswith(".__init__")
            and hasattr(module, "__file__")
            and module.__file__.endswith("__init__.py")
        ):
            global_filename = module_name + ".__init__.py"
        else:
            global_filename = module_name + ".py"
        return global_filename

    def get_assignment_target(
        self,
        target_str: str,
    ):
        """
        Extract the target of the global assignment.
        """
        assert isinstance(
            target_str, str
        ), f"target_str must be a string. got {target_str}"

        target_node = gast.parse(target_str).body[0].value
        if isinstance(target_node, gast.Name):
            return target_node.id
        elif isinstance(target_node, gast.Subscript):
            return target_node.value.id
        else:
            return ast_to_source_code(target_node).strip()

    def _add_globals(self, assign_str, module_str, from_mixed_object_like=False) -> str:
        """
        Process and add global variables to the transformer's global list.

        This method handles the complexities of global variable assignments,
        including caching, cyclic dependency resolution, self-referencials,
        and nesting.

        Args:
            assign_str (str): The assignment string for the global variable.
            module_str (str): The module string where the global is defined.
            from_mixed_object_like (bool): Whether the global is defined in a mixed function (only relevant for ivy functions).
        Returns:
            str: The assignment target or value, depending on whether the global
                 is inlinable or not.

        Raises:
            AssertionError: If multiple targets are found for the assignment.
        """
        assign_target = ""
        # return if the assignment string is empty
        if not assign_str:
            return assign_target
        # return if capturing an internal global which belongs to a builtin module
        if module_str.split(".")[0] in BUILTIN_LIKELY_MODULE_NAMES:
            return assign_target

        assign_node = gast.parse(assign_str).body[0]
        if isinstance(assign_node, gast.Assign):
            assert (
                len(assign_node.targets) == 1
            ), f"multiple targets found for assignment {assign_str}"

        target_node = (
            assign_node.targets[0]
            if isinstance(assign_node, gast.Assign)
            else assign_node.target
        )
        value_node = assign_node.value
        # Get the target variable for the assignment
        assign_target = ast_to_source_code(target_node).strip()

        cached_glob = self.transformer.cacher.globals_cache.get(
            (assign_target, module_str)
        )

        if cached_glob:
            """
            cached_glob_origin: origin object of the cached global
            curr_obj_origin: current object being transformed.
            NOTE: for globals, we consider their parent(ie: origin) as the current object.

            Case 1: cached_glob_obj_like.object_like == current_obj_like.object_like --> simply return the cached global
            Case 2: cached_glob_obj_like.object_like != current_obj_like.object_like --> add the cached global to the current transformer
            """
            # update the context in which the cached global is being used
            if cached_glob.object_like != self.transformer.object_like:
                new_glob = deepcopy(cached_glob)
                new_glob.ctx = self.current_context()
                assert (
                    new_glob.ctx is not None
                ), f"No context found for {cached_glob.assignment_target} in {self.transformr.object_like.filename}"
                self.transformer.globals.append(new_glob)
            else:
                # if the object like is same but if the new context takes greater precedence,
                # then update the context in which the cached global is being used
                if self.current_context().value > cached_glob.ctx.value:
                    cached_glob.ctx = self.current_context()
                    assert (
                        cached_glob.ctx is not None
                    ), f"No context found for {cached_glob.assignment_target} in {self.transformr.object_like.filename}"

            # attach any global dependencies if necessary
            self.postprocess_origin_info_for_globals(cached_glob, assign_target)

            # return assign_value
            if cached_glob.assignment_target in self.variables and isinstance(
                value_node, (gast.Attribute, gast.Call)
            ):
                # inline the global if it belongs to a local variable.
                # eg: pad = torch.pad ; def foo(x,pad): return pad(..)
                # here, `pad` is a local variable and is being used as a global
                # so we will inline: def foo(x,pad): return torch.pad(..)
                return ast_to_source_code(value_node).strip()
            # now simply return the assignment target (ie: the cached global)
            return self.get_assignment_target(cached_glob.assignment_target)

        # Check if the global is already in the stack (potential cyclic dependency)
        stk_glob = next(
            (
                stk_glob
                for stk_glob in self.glob_stack
                if stk_glob.target_str == assign_target
            ),
            None,
        )
        if stk_glob is not None:
            return self.get_assignment_target(stk_glob.target_str)

        # if the global is NOT present in the stack, proceed with transforming the global
        node_to_transform = gast.Expr(value=assign_node.value)

        # add the global to the glob_stack
        stack_obj = StackObjectLike(
            value_node=node_to_transform,
            target_str=assign_target,
            object_like=self.transformer.object_like,
        )
        self.glob_stack.append(stack_obj)

        # attach origin info to the RHS node
        parent_node = (
            self.root.body[0] if isinstance(self.root, gast.Module) else self.root
        )
        origin_info = copy(getattr(parent_node, ORIGI_INFO, None))
        if origin_info:
            # 1) flag this node as belonging from the globals.
            # 2) Also set the origin for the global to be the root global in the current chain.
            # This information is used by the self.process_origin_info method of the BaseRecurser to
            # differentiate a regular object from a global object.
            origin_info.from_global = True
            origin_info.origin_obj = self.glob_stack[0].object_like
            setattr(node_to_transform, ORIGI_INFO, origin_info)

        # determine the module associated with the RHS of the global assignment
        # we need the module so that later transformations (called during transform_node)
        # can retrieve the live object associated with this global and handle it accordingly.
        # NOTE: for ivy globals, we retrieve the module from `ivy.utils.backend.handler.ivy_original_dict`.
        # this is done bcz at this stage of the transformation, we have already set the backend in ivy.
        # hence the modules in ivy will now be pointing to the specific backend set. Whereas certain ivy
        # globals (e.g. ivy.SupportsBufferProtocol) are defined inside the functional ivy module, not the backend one.
        if (
            ivy.backend_stack
            and module_str.startswith("ivy.")
            and "frontends" not in module_str
            and not from_mixed_object_like
        ):
            try:
                module = ivy.utils.backend.handler.ivy_original_dict[
                    module_str.split(".")[-1]
                ]
                try:
                    getattr(module, assign_target)
                except AttributeError:
                    module = get_module(module_str)
            except KeyError:
                module = get_module(module_str)
        else:
            module = get_module(module_str)

        # Visit and potentially modify the right-hand side of the assignment
        transformed_node = self.transformer.transform_node(node_to_transform, module)
        transformed_node_origin_info = getattr(node_to_transform, ORIGI_INFO, None)

        # pop the global from the glob_stack now that the transformation is complete
        stack_obj = self.glob_stack.pop()
        assign_node.value = transformed_node.value

        # 1. create the glob obj and cache it
        global_filename = self.get_global_filename(
            object_like=self.transformer.object_like,
            module=module,
            from_mixed_object_like=from_mixed_object_like,
        )
        if transformed_node_origin_info:
            global_dependencies = transformed_node_origin_info.global_dependencies
        else:
            global_dependencies = {}
        glob_ctx = self.current_context()
        assert (
            glob_ctx is not None
        ), f"No context found for {assign_target} in {self.transformr.object_like.filename}"
        obj = GlobalObjectLike(
            self.transformer.object_like,
            target_node=target_node,
            value_node=assign_node.value,
            global_filename=global_filename,
            global_dependencies=global_dependencies,
            ctx=glob_ctx,
        )
        if origin_info:
            # set the origin for this global to be the root global in the current chain.
            # this helps out in dealing with different scenarios of cached globals.
            obj.origin = origin_info.origin_obj

        # Special Case: check if the global obj represent an ivy global. If so, set the is_ivy_global flag. This is
        # done so that the global can be imported within the module its being used in. The general convention
        # used here is that ivy globals injected inside helpers.py module unlike regular globals which are injected
        # in their parent obj's module.
        if transformed_node_origin_info and transformed_node_origin_info.is_ivy_global:
            obj.is_ivy_global = True

        # 3. cache the global
        self.transformer.cacher.globals_cache.cache(
            (obj.assignment_target, module_str), obj
        )

        # 4. add this to the globals
        self.transformer.globals.append(obj)

        # 5. attach any global dependencies if necessary
        self.postprocess_origin_info_for_globals(obj, assign_target)

        # 6a. return assign_value
        if assign_target in self.variables and isinstance(
            value_node, (gast.Attribute, gast.Call)
        ):
            # inline the global if it belongs to a local variable.
            # eg: pad = torch.pad ; def foo(x,pad): return pad(..)
            # here, `pad` is a local variable and is being used as a global
            # so we will inline: def foo(x,pad): return torch.pad(..)
            return ast_to_source_code(value_node).strip()
        # 6b. return assign_target
        return self.get_assignment_target(assign_target)

    def _match_globals(
        self,
        name_to_search: str,
        module_to_search: types.ModuleType,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Match and retrieve global variable assignments from a given module.

        This method attempts to find the assignment string and module string
        for a given global variable name. It first checks a predefined dictionary
        of Ivy globals (IVY_GLOBS) and then falls back to recursively searching
        the module's source code.

        Args:
            name_to_search (str): The name of the global variable to search for.
            module_to_search (types.ModuleType): The module in which to search for the global variable.

        Returns:
            Tuple[Optional[List[str]], Optional[str]]: A tuple containing:
                - glob_assign_str (Optional[List[str]]): A list of assignment strings for the global variable,
                  or an empty string if not found or if it's in NAME_GLOBS_TO_IGNORE.
                - glob_module_str (Optional[str]): The module string where the global is defined,
                  or None if not found.

        Note:
            - This method prioritizes retrieving globals from the IVY_GLOBS dictionary
              for Ivy-related modules to handle backend-specific issues.
            - If the global is not found in IVY_GLOBS or the module is not Ivy-related,
              it falls back to using the get_global_assignment function to recursively traverse
              the module's source code.
            - Globals listed in NAME_GLOBS_TO_IGNORE are treated as if they were not found.
        """
        # try retrieving the global via:
        # 1) predefined IVY_GLOBS dict
        # 2) recursing into the module's source code
        if (
            module_to_search.__name__.startswith("ivy.")
            and "ivy_" not in module_to_search.__name__
            and name_to_search in IVY_GLOBS
        ):
            glob_assign_str, glob_module_str = IVY_GLOBS[name_to_search]
        else:
            glob_assign_str, glob_module_str = get_global_assignment(
                module_to_search, name_to_search
            )
        all_glob_assign_str = (
            glob_assign_str
            if isinstance(glob_assign_str, (list, tuple))
            else (glob_assign_str,)
        )
        valid_global_assignments = []
        for glob_assign_str in all_glob_assign_str:
            glob_target_str = (
                glob_assign_str.split(" ")[0] if glob_assign_str else glob_assign_str
            )
            if (
                glob_target_str
                and glob_target_str in IVY_STANDARD_GLOBALS_TARGET_TO_MODULE
            ):
                from_obj = glob_target_str
                from_mod = IVY_STANDARD_GLOBALS_TARGET_TO_MODULE[from_obj]
                import_statement = create_relative_import_statement(
                    from_mod=from_mod,
                    from_obj=from_obj,
                    current_module_name=self.transformer.object_like.module,
                )
                _, from_import, _, import_obj = import_statement.split(" ")
                self.transformer._from_imports.add((from_import, import_obj, None))
                glob_assign_str = ""
            elif (
                glob_target_str
                and glob_target_str in FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE
            ):
                from_obj = glob_target_str
                from_mod = FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE[from_obj]
                if from_mod != self.transformer.object_like.module:
                    import_statement = create_relative_import_statement(
                        from_mod=from_mod,
                        from_obj=from_obj,
                        current_module_name=self.transformer.object_like.module,
                    )
                    _, from_import, _, import_obj = import_statement.split(" ")
                    self.transformer._from_imports.add((from_import, import_obj, None))
                glob_assign_str = ""
            else:
                glob_assign_str = (
                    ""
                    if glob_target_str is None
                    or glob_target_str in NAME_GLOBS_TO_IGNORE
                    or "_frontend" in self.transformer._source
                    and glob_target_str in FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE
                    else glob_assign_str
                )
            valid_global_assignments.append(glob_assign_str)

        return valid_global_assignments, glob_module_str

    def _should_transform_global(
        self,
        node,
        target_str: str,
        original_str: str,
        module_to_search: Optional[types.ModuleType],
    ) -> bool:
        """
        Determine if a global variable should be transformed.

        This method implements several checks to guard against transforming globals
        from built-in modules or internal native modules of the source framework.

        Args:
            target_str (str): The name of the global variable.
            original_str (str): The original string representation of the global variable. (eg: `A.B.C` where the C is the target_str)
            module_to_search (types.ModuleType): The module where the global is defined.

        Returns:
            bool: True if the global should be transformed, False otherwise.
        """
        target_obj = (
            None
            if module_to_search is None
            else get_function_from_modules(target_str, [module_to_search])
        )
        # run filters on the target_str, module_to_search, and target_obj
        return (
            self._check_is_target_str_valid(original_str=original_str)
            and self._check_is_module_to_search_valid(module_to_search=module_to_search)
            and self._check_is_target_obj_valid(
                node=node,
                target_obj=target_obj,
                module_to_search=module_to_search,
            )
        )

    def _handle_globals(
        self,
        node: gast.AST,
        target_str: str,
        original_str: str,
        module_to_search: Union[types.ModuleType, List[types.ModuleType]],
    ) -> gast.AST:
        """
        Handle global variables by potentially transforming them.

        Args:
            node (gast.AST): The AST node being processed.
            target_str (str): The name of the global variable.
            module_to_search (Union[types.ModuleType, List[types.ModuleType]]):
                The module(s) where the global might be defined.

        Returns:
            gast.AST: The potentially transformed AST node.
        """

        all_modules = (
            (module_to_search,)
            if not isinstance(module_to_search, (list, tuple))
            else module_to_search
        )
        for module in all_modules:
            if self._should_transform_global(node, target_str, original_str, module):
                assign_strs, glob_module_str = self._match_globals(target_str, module)
                for assign_str in assign_strs:
                    new_global = self._add_globals(assign_str, glob_module_str)
                if new_global:
                    return gast.parse(new_global).body[0].value
        return node

    def _check_is_module_to_search_valid(
        self,
        module_to_search: Optional[types.ModuleType],
    ) -> bool:
        return module_to_search is not None and not any(  # module is accessible
            module_to_search.__name__.startswith(mod)
            for mod in (BUILTIN_LIKELY_MODULE_NAMES + self.unsupported_mods)
        )  # module is not from builtin modules (math, itertools etc.) or unsupported native modules(torch._C, torch.fx etc.)

    def _check_is_target_str_valid(
        self,
        original_str: str,
    ) -> bool:

        target_str = original_str.split(".")[-1]
        return (
            original_str
            not in self.variables  # is not a local variable belonging to the function scope
            and not target_str.startswith(
                "__"
            )  # is not a dunder attribute (eg: `__version__` in `torch.__version__`)
        )

    def _check_is_target_obj_valid(
        self,
        node: gast.AST,
        target_obj: Union[type, types.FunctionType],
        module_to_search: Optional[types.ModuleType],
    ) -> bool:

        # object is not from builtin modules(math, itertools etc.) or unsupported native modules(torch._C, torch.fx etc.)
        is_global_from_builtin_or_unsupported_mod = lambda target_obj: any(
            (
                hasattr(target_obj, "__module__")
                and target_obj.__module__ is not None
                and target_obj.__module__.startswith(mod)
            )
            for mod in (BUILTIN_LIKELY_MODULE_NAMES + self.unsupported_mods)
        )
        # object is from builtin modules(math, itertools etc.) or native modules(torch, tf, etc.) BUT
        # is being used as an alias in module_to_search.
        is_global_from_builtin_or_unsupported_mod_but_is_alias = lambda target_obj: any(
            hasattr(target_obj, "__module__")
            and target_obj.__module__.startswith(mod)
            and are_modules_distinct(
                target_obj.__module__, module_to_search.__name__
            )  # 2 modules are only distinct if module_to_search comes from the left part of an attribute node. (eg: <Mod_a> in attribute Mod_a.foo)
            and not module_to_search.__name__.startswith("ivy.")
            for mod in (BUILTIN_LIKELY_MODULE_NAMES + SUPPORTED_BACKENDS_PREFIX)
        )

        return (
            target_obj is not None  # object is accessible from given module
            and not is_global_from_builtin_or_unsupported_mod(target_obj)
            or is_global_from_builtin_or_unsupported_mod_but_is_alias(target_obj)
        )

    def _handle_mixed_functions(self, object_like: BaseObjectLike) -> BaseObjectLike:
        """
        Handles Ivy Mixed Functions with attributes (`compos` and `partial_mixed_handler`) by
        creating synthetic global assignments and attaching them to the original function.
        """
        if object_like.compos_name:
            # 1. create the assignment str
            object_like_name = NAME_GENERATOR.generate_name(object_like)
            assignment_target = f"inspect.unwrap({object_like_name}).compos"
            assignment_value = f"{object_like.compos_name}"
            assignment_str = f"{assignment_target} = {assignment_value}"

            # 2. transform the global
            self._add_globals(
                assignment_str, object_like.compos_module, from_mixed_object_like=True
            )

            # assert the global was successfully transformed and captured
            if (
                not self.transformer.globals
                or self.transformer.globals[-1].assignment_target != assignment_target
            ):
                raise ValueError(
                    f"Expected global assignment for {assignment_target} not found."
                )

            # 3. update the global to be a bottom global
            compos_glob = self.transformer.globals[-1]
            compos_glob.position = Position.BOTTOM
            compos_glob.is_inlinable = False
            self.transformer.globals[-1] = compos_glob

        if object_like.mixed_condition_source:
            # 1. create the assignment str
            object_like_name = NAME_GENERATOR.generate_name(object_like)
            assignment_target = (
                f"inspect.unwrap({object_like_name}).partial_mixed_handler"
            )
            root = gast.parse(object_like.mixed_condition_source).body[0]
            if isinstance(root, gast.FunctionDef):
                mixed_condition_source = root.name
            else:
                mixed_condition_source = object_like.mixed_condition_source
            assignment_value = f"{mixed_condition_source}"
            assignment_str = f"{assignment_target} = {assignment_value}"

            # 2. transform the global
            self._add_globals(
                assignment_str, object_like.module, from_mixed_object_like=True
            )

            # assert the global was successfully transformed and captured
            if (
                not self.transformer.globals
                or self.transformer.globals[-1].assignment_target != assignment_target
            ):
                raise ValueError(
                    f"Expected global assignment for {assignment_target} not found."
                )

            # 3. update the global to be a bottom global
            mixed_object_like_cond_glob = self.transformer.globals[-1]
            mixed_object_like_cond_glob.position = Position.BOTTOM
            mixed_object_like_cond_glob.is_inlinable = False
            self.transformer.globals[-1] = mixed_object_like_cond_glob

        return object_like
