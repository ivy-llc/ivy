# global
import importlib

# local
from ...transformers.base_transformer import (
    BaseTransformer,
)
from ...transformer import Transformer
import gast
from types import FunctionType, ModuleType
from ivy.transpiler.utils.api_utils import (
    get_function_from_modules,
    is_compiled_module,
    is_ivy_api,
    SUPPORTED_BACKENDS_PREFIX,
)
from ivy.transpiler.utils.ast_utils import (
    ast_to_source_code,
    get_import_dict,
    get_module,
    get_function_vars,
)
from ... import transformer_globals as glob
from ivy.transpiler.utils.conversion_utils import (
    BUILTIN_LIKELY_MODULE_NAMES,
)
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)


class BaseNameCanonicalizer(BaseTransformer):
    """
    A class to rename gast.Name and gast.Call nodes to follow the standard
    "module.<func_name>" format if they are part of the standard modules.
    It also captures any standard imports during the process.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        import_dict, from_import_dict, internal_obj_dict = get_import_dict(
            self.transformer.object_module
        )
        self.import_dict = import_dict
        self.from_import_dict = from_import_dict
        self.internal_obj_dict = internal_obj_dict
        variables, non_locals_and_globals = get_function_vars(self.root)
        self.variables = variables.union(non_locals_and_globals)
        self.visit(self.root)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_str = ast_to_source_code(node).strip()
        node = self._rename_node(node, attr_str)
        return node

    def visit_Name(self, node):
        name_str = ast_to_source_code(node).strip()
        node = self._rename_node(node, name_str)
        return node

    def visit_Call(self, node):
        func_str = ast_to_source_code(node.func).strip()
        node.func = self._rename_node(node, func_str)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        # This method is called for each variable annotation in the tree
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        self.generic_visit(node)
        return node

    def visit_arguments(self, node):
        for arg in node.args:
            self.generic_visit(arg)
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        for arg in node.posonlyargs:
            self.generic_visit(arg)
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        for arg in node.kwonlyargs:
            self.generic_visit(arg)
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)

        if node.vararg and node.vararg.annotation is not None:
            node.vararg.annotation = self.visit(node.vararg.annotation)
        if node.kwarg and node.kwarg.annotation is not None:
            node.kwarg.annotation = self.visit(node.kwarg.annotation)
        self.generic_visit(node)
        return node

    def _rename_node(self, node, func_str):
        native_mods = SUPPORTED_BACKENDS_PREFIX
        # Retrieve the original function
        module = self.transformer.object_module
        orig_func = get_function_from_modules(func_str, module)
        left_mod = ".".join(func_str.split(".")[:-1])
        try:
            bool(orig_func)
        except (ValueError, RuntimeError):
            return node.func if isinstance(node, gast.Call) else node

        # only canonicalize if:
        # 1) func_str does NOT represent a local variable
        # 2) func_str does NOT represent a cls inside the glob.CLASSES_TO_IGNORE
        # 3) func_str represents a function that is not inside ivy api
        if (
            orig_func is not None
            and func_str not in self.variables
            and not (
                isinstance(orig_func, ModuleType)
                and orig_func.__name__.startswith("ivy.")
            )
        ):
            in_import_dict = func_str in self.import_dict
            in_from_import_dict = func_str in self.from_import_dict
            in_internal_object_dict = func_str in self.internal_obj_dict
            if in_import_dict or in_from_import_dict:
                dict_to_key = (
                    self.import_dict if in_import_dict else self.from_import_dict
                )
                import_obj = dict_to_key[func_str]
                if (
                    any(
                        import_obj.module.startswith(prefix)
                        for prefix in BUILTIN_LIKELY_MODULE_NAMES
                    )
                    or is_compiled_module(import_obj.module)
                    or func_str in glob.CLASSES_TO_IGNORE
                ):
                    # case 1. following objects aren't canonicalized, they are just captured.
                    # i) builtin imports
                    # ii) compiled modules
                    # iii) classes in glob.CLASSES_TO_IGNORE
                    (
                        self.transformer._from_imports.add(
                            (import_obj.module, import_obj.obj, import_obj.asname)
                        )
                        if in_from_import_dict
                        else self.transformer._imports.add(
                            (import_obj.module, import_obj.asname)
                        )
                    )
                elif any(
                    import_obj.module.startswith(prefix)
                    for prefix in native_mods + ["ivy"]
                ):
                    # case 2. native + ivy imports aren't captured, they are just canonicalized.
                    return gast.parse(import_obj.canonical_name).body[0].value
                else:
                    # case 3. custom imports are both captured and canonicalized.
                    # we add the module captured to the transformer.object_module. This
                    # is done so that later transformations can correctly retrieve any objects associated
                    # with this module and make informed decisions based on that.
                    module = get_module(import_obj.module)
                    if module and module not in self.transformer.object_module:
                        self.transformer.object_module = module
                    # case 4. handle edge cases along the lines of `kornia.utils.one_hot.one_hot`
                    # where the a part of the canonical_name can be used to retrieve the original
                    # func from the current module e.g. `kornia.utils.one_hot` actually points to
                    # the func like object and doing `kornia.utils.one_hot.one_hot` just fails in
                    # the interpreter because of the way these module namespace have been defined in
                    # kornia, in these cases if we are able to retrieve the same func obj with a part of
                    # the canonical name, just use that canonical name as the return value
                    retrieved_func = get_function_from_modules(
                        import_obj.module, self.transformer.object_module
                    )
                    if retrieved_func is orig_func:
                        return gast.parse(import_obj.module).body[0].value

                    # case 5. handle edge cases where the module part of the canonical name references both a
                    # module and function, in this case the function should (probably) be retrieved from the
                    # parent module
                    if isinstance(orig_func, (FunctionType, type)):
                        try:
                            parent_module = import_obj.module.rsplit(".", 1)[0]
                            mod_list = [importlib.import_module(parent_module)]
                            retrieved_func = get_function_from_modules(
                                import_obj.obj, mod_list
                            )
                            if retrieved_func is orig_func:
                                return (
                                    gast.parse(parent_module + "." + import_obj.obj)
                                    .body[0]
                                    .value
                                )
                        except ImportError:
                            pass

                    return gast.parse(import_obj.canonical_name).body[0].value

            elif left_mod in BUILTIN_LIKELY_MODULE_NAMES:
                # case 4. already canonicalized builtin functions are captured. No need to canonicalize them again.
                self.transformer._imports.add((left_mod, None))
            elif in_internal_object_dict:
                dict_to_key = self.internal_obj_dict
                import_obj = dict_to_key[func_str]

                # only canonicalize internal objects(ie: objects which are defined within the same module rather than being imported)
                # if they belong to the set of builtin modules (eg: math, os, etc..)
                if any(
                    import_obj.module.startswith(prefix)
                    for prefix in BUILTIN_LIKELY_MODULE_NAMES
                ):
                    self.transformer._imports.add((import_obj.module, None))
                    return gast.parse(import_obj.canonical_name).body[0].value

        return node.func if isinstance(node, gast.Call) else node
