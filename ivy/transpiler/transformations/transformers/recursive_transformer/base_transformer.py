# global
from abc import abstractmethod, ABC
import gast
import inspect
import os
import types

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....translations.translator import Translator
from ....utils.ast_utils import (
    ast_to_source_code,
    extract_target_object_name,
    get_function_vars,
    get_module,
    TranslatedContext,
)
from ....utils.conversion_utils import is_builtin_function
from ....utils.api_utils import (
    TRANSLATED_OBJ_SUFFIX,
    is_ivy_api,
    has_conv_args,
)
from ....utils.decorator_utils import (
    handle_methods,
    handle_get_item,
    handle_set_item,
    load_state_dict_from_url,
    dummy_inplace_update,
    handle_transpose_in_input_and_output,
    handle_transpose_in_input_and_output_for_functions,
    handle_transpose_in_pad,
    store_config_info,
)
from ..base_transformer import (
    BaseTransformer,
)
from ..rename_transformer import (
    BaseRenameTransformer,
)
from ....utils.naming_utils import NAME_GENERATOR
from ....utils.origin_utils import ORIGI_INFO
from ....utils.type_utils import Types
from ... import transformer_globals as glob


def has_same_code(obj, orig_obj):
    obj, orig_obj = inspect.unwrap(obj), inspect.unwrap(orig_obj)
    return (
        hasattr(obj, "__code__")
        and hasattr(orig_obj, "__code__")
        and obj.__code__ == orig_obj.__code__
    )


class BaseRecurser(BaseTransformer, ABC):
    """
    The `RecursiveTransformer` is an abstract base class that provides a framework for
    recursively transforming AST nodes. It defines the common structure and operations
    for the transformation process. Specific transformations are implemented in the
    child classes.
    """

    _instance = None
    call_stack = []
    _metaclasses = []  # adding this to handle metaclass issues for JAX backend

    def __new__(cls, *args, **kwargs):
        # Create a new instance and store it in the class variable
        BaseRecurser._instance = super(BaseRecurser, cls).__new__(cls)
        return BaseRecurser._instance

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.function_or_class_stack = []
        self.profiling = False
        self.context_stack = [TranslatedContext.VARIABLE]

    def transform(self):
        variables, non_locals_and_globals = get_function_vars(self.root)
        self.variables = variables.union(non_locals_and_globals)
        self.visit(self.root)
        if self.transformer.configuration.profiler and self.profiling:
            self.transformer.configuration.profiler(
                self.root, self.transformer, self.configuration
            ).visit(self.root)

    @classmethod
    def _get_instance(cls):
        """
        Method to return the already initialized instance.
        """
        return cls._instance

    @classmethod
    def simple_transform(cls, obj_name: str, obj_to_translate):
        return cls._get_instance()._recursively_translate(
            obj_name=obj_name,
            obj_to_translate=obj_to_translate,
        )

    def postprocess_origin_info(self, object_like_to_translate, new_name: str):
        """
        attaches the current object as a dependency to it's parent.
        NOTE: this version is specific for handling dependency cases when the RHS object is a
        callable (ie: function/class). For globals, see `postprocess_origin_info_for_globals`
        in the BaseGlobalsTransformer.

        Eg: if we have the following code:
        def foo(): pass # inside module A
        Glob = {'f': foo} # inside module B

        when we translate the `foo` object, we want to attach the `foo` object as a dependency to the `Glob` object (ie: its parent).
        This will allow us to correctly add necessary imports to the translated code.

        Eg: In the translated code, we will have the following code:
        # module_B.py
        from module_A import foo
        Glob = {'f': foo}
        """
        parent_node = (
            self.root.body[0] if isinstance(self.root, gast.Module) else self.root
        )
        origin_info = getattr(parent_node, ORIGI_INFO, None)
        if origin_info and origin_info.from_global:
            origin_info.global_dependencies[new_name] = (
                object_like_to_translate.filename
            )
            setattr(parent_node, ORIGI_INFO, origin_info)

    def preprocess_origin_info(self, obj_to_translate, new_name):
        """
        Process the origin information of the current object to determine its handling.

        Args:
            obj_to_translate (Any): The current object being translated.
            new_name (str): The translated name of the object.

        Returns:
            tuple: A tuple containing from_global (bool) and parent (object).
                from_global (bool): Indicates whether the object comes from a global statement.
                parent (object): The parent object to use for emitting the global.

        The checks serve the following purposes:
            1) origin_info.from_global: Checks whether the current object comes from a global statement,
            e.g., (GLOB = Foo()) where obj_to_translate= Foo.

            2) is_ivy_api(obj_to_translate): Checks whether the current object is a custom object
            or from the ivy API (e.g., GLOB = ivy.mean --> False, GLOB = Foo() --> True).

            This prevents emitting ivy functions inside other modules and instead always emits them inside helpers.py.
            3) Checks whether the current object is a translated version from the ivy API,
            e.g., GLOB = Translated_torch_frnt, GLOB = ivy_mean_bknd.
        """
        parent_node = (
            self.root.body[0] if isinstance(self.root, gast.Module) else self.root
        )
        origin_info = getattr(parent_node, ORIGI_INFO, None)

        if origin_info and origin_info.from_global:
            is_ivy_glob = is_ivy_api(obj_to_translate) or any(
                new_name.endswith(substr) for substr in TRANSLATED_OBJ_SUFFIX
            )
            if is_ivy_glob:
                # Add information to the node to indicate that this is an ivy API global.
                # This means we will emit the ivy API global inside the helpers.py module
                # but now we also need to cater for dependency import.
                origin_info.is_ivy_global = True
                setattr(parent_node, ORIGI_INFO, origin_info)
                from_global = False
                parent = self.transformer.object_like
            else:
                from_global = True
                parent = (
                    origin_info.origin_obj
                )  # Get the parent info from origin_info. This information is used to emit the global within its parent's module.
        else:
            from_global = False
            parent = self.transformer.object_like

        return from_global, parent

    def is_var(self, node, name):
        return name in self.variables

    def _recursively_translate(self, obj_name: str, obj_to_translate, node=None):
        # 1 associate a unique name with the object
        new_name = NAME_GENERATOR.generate_name(obj_to_translate)

        # 2 process the origin info of this object to determine
        # whether its from a global and if so, what is its parent
        from_global, parent = self.preprocess_origin_info(obj_to_translate, new_name)

        # 3 Recursively transform the function
        from ivy.transpiler.translations.data.object_like import (
            BaseObjectLike,
        )

        ctx = self.current_context()
        current_depth = Translator.depth
        Translator.depth += 1
        object_like_to_translate = BaseObjectLike.from_object(
            obj=obj_to_translate,
            parent=parent,
            root_obj=None,
            from_global=from_global,
            ctx=ctx,
            depth=Translator.depth,
            target=self.transformer.target,
            base_output_dir=self.transformer.configuration.base_output_dir,
        )
        self.postprocess_origin_info(object_like_to_translate, new_name)
        # avoid self-translations if obj_to_translate:
        # 1) is the same obj as the one we are translating OR
        # 2) shares the same code as the object we are translating OR
        # 3) is in the call stack
        if not self.transformer.object_like.is_same_object(obj_to_translate) and all(
            [
                object_like_to_translate not in self.call_stack,
                not any(
                    object_like_to_translate.has_same_code(obj_like)
                    for obj_like in self.call_stack
                ),
            ]
        ):

            self.call_stack.append(object_like_to_translate)

            _ = Translator.simple_translate(
                object_like=object_like_to_translate,
                depth=Translator.depth,
                parent=parent,
                from_global=from_global,
                configuration=self.transformer.configuration,
                cacher=self.transformer.cacher,
                logger=self.transformer.logger,
                output_dir=self.transformer.output_dir,
                profiling=self.profiling,
                reuse_existing=self.transformer.reuse_existing,
            )
            self.call_stack.pop()

        if object_like_to_translate in self.call_stack:
            # self referencing loops
            # eg: A --> B --> A . A references B ; B references A.
            # `A` will get added as a local import inside B. This is to avoid
            # circular imports where 2 modules import each other.
            self.transformer.circular_ref_object_likes.add(object_like_to_translate)

        # 4 Replace the original name with the translated name in the AST
        BaseRenameTransformer(self.root).rename(old_name=obj_name, new_name=new_name)

        # reset the depth
        Translator.depth = current_depth
        # 5 Replace the node with a new Name node using the translated name
        if node is not None:
            return gast.Name(
                id=new_name, ctx=node.ctx, type_comment=None, annotation=None
            )

    def _maybe_recursively_translate(self, orig_obj, name_str, node):
        # do not translate if:
        # 1) the name belongs to a local variable
        # 2) the name belongs to a global variable
        if self.is_var(node, name_str) or any(
            name_str == globs.assignment_target for globs in self.transformer.globals
        ):
            return node

        obj_to_translate = orig_obj
        obj_name = name_str
        is_obj_unsupported = lambda obj: obj.__name__ in glob.CLASSES_TO_IGNORE
        # Check if the object should be translated
        if self._should_translate(
            obj_name, obj_to_translate
        ) and not is_obj_unsupported(obj_to_translate):
            # Proceed to recursively translate the object
            return self._recursively_translate(
                obj_name=obj_name,
                obj_to_translate=obj_to_translate,
                node=node,
            )

        return node

    def _handle_name_or_attribute(self, node):
        # Convert the node back to source code and get the corresponding object
        name_str = ast_to_source_code(node).strip()
        module = self.transformer.object_module

        if name_str in glob.CLASSES_TO_IGNORE:
            return node

        if (
            name_str == "handle_get_item"
        ):  # TODO: generalize this if there are more such internal functions being utilized.
            orig_obj = handle_get_item
        elif name_str == "handle_set_item":
            orig_obj = handle_set_item
        elif name_str in ("handle_methods", "ivy_handle_methods"):
            orig_obj = handle_methods
        elif name_str in (
            "load_state_dict_from_url",
            "ivy_load_state_dict_from_url_frnt",
        ):
            orig_obj = load_state_dict_from_url
        elif (
            name_str in ("ivy.inplace_update", "ivy_inplace_update")
            and self.transformer.target == "jax"
        ):
            orig_obj = dummy_inplace_update
        elif name_str == "handle_transpose_in_input_and_output":
            orig_obj = handle_transpose_in_input_and_output
        elif name_str == "handle_transpose_in_input_and_output_for_functions":
            orig_obj = handle_transpose_in_input_and_output_for_functions
        elif name_str == "handle_transpose_in_pad":
            orig_obj = handle_transpose_in_pad
        elif name_str == "store_config_info":
            orig_obj = store_config_info
        else:
            if name_str == "kornia.feature.loftr.resnet_fpn.ResNetFPN_8_2":
                ss = 10
            orig_obj = self._get_function(name_str, module, node)

        return self._maybe_recursively_translate(orig_obj, name_str, node)

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

    def visit_ClassDef(self, node):
        # Iterate over the bases of the class
        self.push_context(TranslatedContext.BASE)
        for i, base in enumerate(self.transformer.object_like.bases):

            # TODO: Remove "ivy.stateful.module" from here when doing
            # stateful transformation from `ivy` to target.
            if (
                base.__module__
                in (
                    "builtins",
                    "enum",
                    "ivy.stateful.module",
                )
                or is_builtin_function(base)
                or base.__name__ in glob.CLASSES_TO_IGNORE
            ):
                continue

            # 1 associate a unique name with the object
            base_new_name = NAME_GENERATOR.generate_name(base)

            # 2 process the origin info of this object to determine
            # whether its from a global and if so, what is its parent
            from_global, parent = self.preprocess_origin_info(base, base_new_name)

            # 3 Recursively transform the function
            from ivy.transpiler.translations.data.object_like import (
                BaseObjectLike,
            )

            ctx = self.current_context()
            current_depth = Translator.depth
            Translator.depth += 1
            object_like_to_translate = BaseObjectLike.from_object(
                obj=base,
                parent=parent,
                root_obj=None,
                from_global=from_global,
                ctx=ctx,
                depth=Translator.depth,
                target=self.transformer.target,
                base_output_dir=self.transformer.configuration.base_output_dir,
            )
            self.postprocess_origin_info(object_like_to_translate, base_new_name)

            is_base_in_call_stack = lambda object_like_to_translate: any(
                [
                    object_like_to_translate in self.call_stack,
                    any(
                        object_like_to_translate.has_same_code(obj_like)
                        for obj_like in self.call_stack
                    ),
                ]
            )
            if is_base_in_call_stack(object_like_to_translate):
                continue

            # Check if the base class is an instance of torch.nn.Module
            if self._is_base_class(base):
                # Replace it with frontend nn.Module
                node.bases[i] = self._get_module_node()
                base_new_name = self._get_module_name()

            else:
                # 2 Recursively transform the base class
                self.call_stack.append(object_like_to_translate)
                Translator.simple_translate(
                    object_like=object_like_to_translate,
                    depth=Translator.depth,
                    parent=parent,
                    from_global=from_global,
                    configuration=self.transformer.configuration,
                    cacher=self.transformer.cacher,
                    logger=self.transformer.logger,
                    output_dir=self.transformer.output_dir,
                    profiling=self.transformer.configuration.profiling,
                    reuse_existing=self.transformer.reuse_existing,
                )
                self.call_stack.pop()
            # 3 rename all gast.Name node's with the new translated version
            old_base_name = ast_to_source_code(node.bases[i]).strip()
            BaseRenameTransformer(self.root).rename(
                old_name=old_base_name, new_name=base_new_name
            )

            # reset the current depth
            Translator.depth = current_depth

        self.pop_context()
        _new_name = NAME_GENERATOR.generate_name(self.transformer.object_like)
        # rename all gast.Name node's with the new translated version
        BaseRenameTransformer(self.root).rename(old_name=node.name, new_name=_new_name)

        # Handle bases
        self.push_context(TranslatedContext.BASE)
        for base in node.bases:
            self.visit(base)
        self.pop_context()

        # Handle keywords
        self.push_context(TranslatedContext.BASE)
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                self._metaclasses.append(ast_to_source_code(keyword.value).strip())

            self.visit(keyword)
        self.pop_context()

        # Handle decorators
        self.push_context(TranslatedContext.DECORATOR)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.pop_context()

        # Visit the rest of the class body
        self.function_or_class_stack.append(node.name)
        self.push_context(TranslatedContext.CLASS_ATTRIBUTE)
        for item in node.body:
            self.visit(item)
        self.pop_context()

        # Pop the function name from the stack
        self.function_or_class_stack.pop()

        if node.name in self._metaclasses and self.transformer.target == "jax":
            # Inherit from `ModuleMeta` to prevent metaclass conflicts, as `nnx.Module`
            # also derives from `ModuleMeta`, which is a metaclass.
            node.bases.insert(0, gast.Name(id="ModuleMeta", ctx=gast.Load()))
        # Rename the function only if it's a top-level class
        if (
            len(self.function_or_class_stack) == 0
            and self.transformer.object_like.type == Types.ClassType
            and NAME_GENERATOR.get_name(self.transformer.object_like)
        ):
            node.name = _new_name
        return node

    def visit_FunctionDef(self, node):
        # translate all function decorators
        new_decorator_list = []
        # check for conflicting methods
        if self.transformer.object_like.name in glob.CONFLICTING_METHODS:
            func_name = self.transformer.object_like.name
            handler_decorator = (
                "handle_set_item"
                if func_name == "set_item"
                else "handle_get_item" if func_name == "get_item" else "handle_methods"
            )
            new_decorator_list.append(gast.parse(handler_decorator).body[0].value)
        # check for transpose optimization
        elif (
            os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
            and any(
                substr == extract_target_object_name(self.transformer.object_like.name)
                for substr in glob.CONV_BLOCK_FNS
            )
            and self.transformer.source == "ivy"
            and node.name == "forward"
        ):
            handler_decorator = "handle_transpose_in_input_and_output"
            new_decorator_list.append(gast.parse(handler_decorator).body[0].value)
        elif (
            os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
            and has_conv_args(node.args)
            and self.transformer.source == "ivy"
            and self.transformer.object_like.is_ivy_api
        ):
            handler_decorator = "handle_transpose_in_input_and_output_for_functions"
            new_decorator_list.append(gast.parse(handler_decorator).body[0].value)
        elif (
            os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
            and self.transformer.source == "ivy"
            and self.transformer.object_like.is_backend_api
            and node.name == "pad"
        ):
            handler_decorator = "handle_transpose_in_pad"
            new_decorator_list.append(gast.parse(handler_decorator).body[0].value)
        elif (
            self.transformer.source == "ivy"
            and node.name == "__init__"
            and not self.transformer.object_like.is_root_obj
        ):
            handler_decorator = "store_config_info"
            new_decorator_list.append(gast.parse(handler_decorator).body[0].value)

        # Handle augmented decorators
        self.push_context(TranslatedContext.DECORATOR)
        for i, decor in enumerate(new_decorator_list):
            new_decor = self.visit(decor)
            new_decorator_list[i] = new_decor
        self.pop_context()

        # Handle existing decorators
        self.push_context(TranslatedContext.DECORATOR)
        for decor in node.decorator_list:
            new_decor = self.visit(decor)
            new_decorator_list.append(new_decor)
        self.pop_context()

        node.decorator_list = new_decorator_list

        # Handle function arguments and type annotations
        self.push_context(TranslatedContext.FUNCTION_ARGS)
        self.visit(node.args)
        self.pop_context()

        # Handle the return type if available
        if node.returns:
            self.push_context(TranslatedContext.TYPE_SPEC)
            self.visit(node.returns)
            self.pop_context()

        # Push the function name onto the stack
        self.function_or_class_stack.append(node.name)

        # Visit the rest of the function body
        self.push_context(TranslatedContext.VARIABLE)
        for item in node.body:
            self.visit(item)
        self.pop_context()

        # Pop the function name from the stack
        self.function_or_class_stack.pop()

        # Rename the function only if it's a top-level function
        if (
            len(self.function_or_class_stack) == 0
            and self.transformer.object_like.type == Types.FunctionType
            and NAME_GENERATOR.get_name(self.transformer.object_like)
        ):
            node.name = NAME_GENERATOR.get_name(self.transformer.object_like)

        return node

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

    def visit_Name(self, node):
        node = self._handle_name_or_attribute(node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node = self._handle_name_or_attribute(node)
        return node

    @abstractmethod
    def _is_base_class(self, base):
        pass

    @abstractmethod
    def _get_module_node(self):
        pass

    @abstractmethod
    def _get_module_name(self):
        pass

    @abstractmethod
    def _get_function(self, name_str, module, node):
        pass

    @abstractmethod
    def _should_translate(self, func_str, orig_func):
        pass
