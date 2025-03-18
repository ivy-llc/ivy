# global
from abc import ABC, abstractmethod
import gast
import os
from typing import Dict, List, Tuple

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)
from ....utils.ast_utils import (
    ast_to_source_code,
    create_relative_import_statement,
    _inject_local_imports_in_function,
    _inject_local_imports_in_class,
    TranslatedContext,
)
from ....utils.naming_utils import NAME_GENERATOR
from ....utils.type_utils import Types


# Base class for converting PyTorch layers to various frameworks
class PytorchToFrameworkLayer(BaseTransformer, ABC):
    """
    Base class for converting PyTorch layers to other frameworks.
    Child classes will define the framework-specific layer mappings and conversions.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
    ) -> None:
        self.root = root
        self.transformer: Transformer = transformer
        self.configuration: BaseTransformerConfig = configuration
        self.layer_mapping: Dict[str, callable] = self.get_layer_mapping()
        self.alias_layer_mapping: Dict[str, Tuple[callable, TranslatedContext]] = {}
        self.local_import_nodes: List[gast.ImportFrom] = []
        self.context_stack: List[TranslatedContext] = [TranslatedContext.VARIABLE]

        if os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", "true") == "true":
            self.data_format = "channels_last"
        else:
            self.data_format = "channels_first"

    @abstractmethod
    def get_layer_mapping(self):
        pass

    @abstractmethod
    def get_name_mapping(self):
        pass

    @abstractmethod
    def convert_conv2d(self, node, is_alias=False):
        pass

    @abstractmethod
    def convert_linear(self, node, is_alias=False):
        pass

    @abstractmethod
    def convert_batchnorm2d(self, node, is_alias=False):
        pass

    @abstractmethod
    def convert_maxpool(self, node, is_alias=False):
        pass

    @abstractmethod
    def get_import_module(self):
        pass

    def transform(self):
        if os.environ.get("USE_NATIVE_LAYERS", "true") == "true":
            self.visit(self.root)
            self.inject_local_imports(self.root)

    def inject_local_imports(self, node):
        if self.transformer.object_like.type == Types.FunctionType:
            func_name = NAME_GENERATOR.get_name(self.transformer.object_like)
            _inject_local_imports_in_function(self.local_import_nodes, func_name, node)
        else:
            _inject_local_imports_in_class(self.local_import_nodes, node)

    def create_local_import(self, node, layer_name):
        import_statement = create_relative_import_statement(
            from_mod=self.get_import_module(),
            from_obj=layer_name,
            current_module_name=self.transformer.object_like.module,
        )
        from_mod = import_statement.split(" ")[1]
        is_compile_time_object = self.current_context() != TranslatedContext.VARIABLE

        if is_compile_time_object:
            # add as module import
            self.transformer._from_imports.add((from_mod, layer_name, None))
        elif all(layer_name != imp.names[0].name for imp in self.local_import_nodes):
            # add the import as a local import
            import_node = gast.ImportFrom(
                module=from_mod,
                names=[gast.alias(name=layer_name, asname=None)],
                level=0,
            )
            self.local_import_nodes.append(import_node)

        return node

    def convert_args_to_kwargs(self, node, args_order):
        args_kwargs = {arg: value for arg, value in zip(args_order, node.args)}
        args_kwargs.update({kw.arg: kw.value for kw in node.keywords})
        return args_kwargs

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

        # Visit default values (if present)
        name_mapping = self.get_name_mapping()
        for idx, default in enumerate(node.defaults):
            if default:
                # Check if the default value is a known layer (e.g., tensorflow_Conv2d)
                if isinstance(default, gast.Name) and default.id in name_mapping:
                    # Store the argument name (from node.args[idx]) as an alias
                    if node.args[0].id in ["self", "cls"]:
                        arg_name = node.args[idx + 1].id
                    else:
                        arg_name = node.args[idx].id
                    _, map_fn = name_mapping[default.id]
                    self.alias_layer_mapping[arg_name] = (
                        map_fn,
                        TranslatedContext.FUNCTION_ARGS,
                    )

                self.push_context(TranslatedContext.FUNCTION_ARGS)
                self.visit(default)
                self.pop_context()

        # Visit keyword-only arguments with default values
        for idx, default in enumerate(node.kw_defaults):
            if default:
                if isinstance(default, gast.Name) and default.id in name_mapping:
                    # Store the kwarg name (from node.kwonlyargs[idx]) as an alias
                    if node.args[0].id in ["self", "cls"]:
                        arg_name = node.kwonlyargs[idx + 1].id
                    else:
                        arg_name = node.kwonlyargs[idx].id
                    _, map_fn = name_mapping[default.id]
                    self.alias_layer_mapping[arg_name] = (
                        map_fn,
                        TranslatedContext.FUNCTION_ARGS,
                    )

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

    def visit_Assign(self, node):
        self.generic_visit(node)
        target = ast_to_source_code(node.targets[0]).strip()
        try:
            map_fn, context = self.alias_layer_mapping[""]
            self.alias_layer_mapping.pop("")
            self.alias_layer_mapping[target] = (map_fn, context)
        except KeyError:
            pass
        return node

    def visit_Name(self, node):
        name_mapping = self.get_name_mapping()
        if node.id in name_mapping:
            new_name, map_fn = name_mapping[node.id]
            self.alias_layer_mapping[""] = (map_fn, TranslatedContext.VARIABLE)
            node.id = new_name

            layer_name = new_name
            import_statement = create_relative_import_statement(
                from_mod=self.get_import_module(),
                from_obj=layer_name,
                current_module_name=self.transformer.object_like.module,
            )
            from_mod = import_statement.split(" ")[1]
            is_compile_time_object = (
                self.current_context() != TranslatedContext.VARIABLE
            )

            if is_compile_time_object:
                # add as module import
                self.transformer._from_imports.add((from_mod, layer_name, None))
            elif all(
                layer_name != imp.names[0].name for imp in self.local_import_nodes
            ):
                # add the import as a local import
                import_node = gast.ImportFrom(
                    module=from_mod,
                    names=[gast.alias(name=layer_name, asname=None)],
                    level=0,
                )
                self.local_import_nodes.append(import_node)

        return node

    def visit_Call(self, node):
        if isinstance(node.func, gast.Name) and node.func.id in self.layer_mapping:
            return self.layer_mapping[node.func.id](node)
        elif (
            isinstance(node.func, gast.Name)
            and node.func.id in self.alias_layer_mapping
        ):
            map_fn, context = self.alias_layer_mapping[node.func.id]
            self.pop_context()
            self.push_context(context)
            return map_fn(node, is_alias=True)
        return self.generic_visit(node)
