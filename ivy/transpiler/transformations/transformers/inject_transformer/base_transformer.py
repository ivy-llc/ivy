# global
import gast
import inspect
import textwrap

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils import pickling_utils
from ....utils.ast_utils import (
    ast_to_source_code,
    is_super_call_node,
)
from ....utils.origin_utils import ORIGI_INFO
from ..base_transformer import (
    BaseTransformer,
)
from ..rename_transformer.base_transformer import (
    BaseRenameTransformer,
)


class BaseSuperMethodsInjector(BaseTransformer):
    """
    Injects methods of the base class of `cls` (which should be a stateful frontend
    class e.g. `torch.nn.Module`), into `cls`, which are not overridden by `cls` in
    its definition, reloads the module for `cls` and returns the transformed and reloaded
    `cls` with methods injected from the base stateful class. This also handles any new
    imports needed to be added as a consequence of the new methods injected.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self._method_nodes_to_add = list()

    def transform(self):
        # If super class methods were already injected for this object, return early
        if self.transformer.cacher.super_methods_injected_cls_object_ids_cache.exist(
            pickling_utils.get_object_hash(self.transformer.object_like)
        ):
            return
        self.visit(self.root)
        self.transformer.cacher.super_methods_injected_cls_object_ids_cache.cache(
            pickling_utils.get_object_hash(self.transformer.object_like)
        )

    def visit_Call(self, node):
        # Check if we have a super call here
        if is_super_call_node(node.func) and node.func.attr in ("__init__",):
            # If the current class is a subclass of frontend
            # torch.nn.Module, proceed to inject methods into it.
            base_class_index = self.transformer.object_like.base_class_index
            if base_class_index != -1:
                # If so, inject that super methods into the current
                # class and update all references to super to point
                # to these injected methods

                return self._inject_super_method_call(
                    node, self.transformer.object_like.bases[base_class_index]
                )
        node = self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        # If the current class is a subclass of frontend
        # torch.nn.Module, proceed to inject methods into it.
        if self.transformer.object_like.base_class_index != -1:
            # Retrieve all the members of the current class
            # that are from `torch.nn.Module`
            base_methods = self.transformer.object_like.base_methods
            if base_methods:
                # Retrieve the module of the base cls e.g.
                # `ivy.functional.frontends.torch.nn.module`
                module_source = self.transformer.object_like.base_module_source

                # Check if module_tree can be fetched from cache
                if self.transformer.cacher.code_to_ast_cache.exist(module_source):
                    module_tree = self.transformer.cacher.code_to_ast_cache.get(
                        module_source
                    )
                else:
                    module_tree = gast.parse(textwrap.dedent(module_source))
                    self.transformer.cacher.code_to_ast_cache.cache(
                        module_source, module_tree
                    )

                # Iterate over all the base cls methods and append for injecting later
                for method_name, method_source in base_methods.items():
                    # First check if we already have the node for this base method in our cache
                    if self.transformer.cacher.code_to_ast_cache.exist(method_source):
                        method_tree = self.transformer.cacher.code_to_ast_cache.get(
                            method_source
                        )
                    else:
                        method_tree = gast.parse(textwrap.dedent(method_source))
                        self.transformer.cacher.code_to_ast_cache.cache(
                            method_source, method_tree
                        )

                    method_node = method_tree.body[0]

                    # Create a new FunctionDef node to be injected later
                    new_method_node = gast.FunctionDef(
                        name=method_name,
                        args=method_node.args,
                        body=method_node.body,
                        decorator_list=method_node.decorator_list,
                        returns=method_node.returns,
                        type_comment=method_node.type_comment,
                        type_params=method_node.type_params if hasattr(method_node, "type_params") else [],
                    )
                    new_method_node = gast.copy_location(new_method_node, method_node)
                    BaseRenameTransformer(new_method_node).rename(
                        old_name="Module", new_name="ivy.Module"
                    )
                    self._method_nodes_to_add.append(new_method_node)

                # Also update the base class to now be ivy.Module instead
                node.bases[self.transformer.object_like.base_class_index] = (
                    gast.Attribute(
                        value=gast.Name(
                            id="ivy",
                            ctx=gast.Load(),
                            annotation=None,
                            type_comment=None,
                        ),
                        attr="Module",
                        ctx=gast.Load(),
                    )
                )

        node = self.generic_visit(node)
        return node

    def visit_Module(self, node):
        node = self.generic_visit(node)

        if not isinstance(node.body[0], gast.ClassDef):
            return node

        # Retrieve the class node
        cls_node = node.body[0]

        # Sort the method nodes to bring any `super..` transformed nodes to the top
        method_nodes_to_add = sorted(
            self._method_nodes_to_add,
            key=lambda x: x.name.startswith("super"),
            reverse=True,
        )
        # Create and return a new ClassDef node
        new_node = gast.ClassDef(
            name=cls_node.name,
            bases=cls_node.bases,
            keywords=cls_node.keywords,
            body=[*cls_node.body, *method_nodes_to_add],
            decorator_list=cls_node.decorator_list,
            type_params=cls_node.type_params if hasattr(cls_node, "type_params") else [],
        )
        new_node = gast.copy_location(new_node, cls_node)
        # attach origin info to the new node
        origin_info = getattr(self.root.body[0], ORIGI_INFO)
        setattr(new_node, ORIGI_INFO, origin_info)
        # Add the new ClassDef node to the module
        node.body.clear()
        node.body.append(new_node)

        return gast.fix_missing_locations(node)

    def _inject_super_method_call(self, node, base):
        base_method_name = node.func.attr
        base_method = getattr(base, base_method_name)
        method_source = inspect.getsource(base_method)

        # First check if we already have the node for this base method in our cache
        if self.transformer.cacher.code_to_ast_cache.exist(method_source):
            method_tree = self.transformer.cacher.code_to_ast_cache.get(method_source)
        else:
            method_tree = gast.parse(textwrap.dedent(method_source))
            self.transformer.cacher.code_to_ast_cache.cache(method_source, method_tree)

        method_node = method_tree.body[0]

        # modify the method_node's body where super().__init__(self, ...) --> super().__init__(...)
        # this is done because self is already passed implicity in python's object model. Hence, passing this explicitly results in
        # `self` being stored in self._args. This leads to issues with `self` forming part of self._layers
        # resulting in an infinite recursion problems during initialization and/or forward pass.

        def remove_explicit_self_from_super_init(node):
            for stmt in node.body:
                if isinstance(stmt, gast.Expr) and isinstance(stmt.value, gast.Call):
                    call = stmt.value
                    if (
                        isinstance(call.func, gast.Attribute)
                        and isinstance(call.func.value, gast.Call)
                        and isinstance(call.func.value.func, gast.Name)
                        and call.func.value.func.id == "super"
                        and call.func.attr == "__init__"
                    ):
                        # Remove 'self' argument if it is the first argument
                        if (
                            len(call.args) > 0
                            and isinstance(call.args[0], gast.Name)
                            and call.args[0].id == "self"
                        ):
                            call.args.pop(0)

            return node

        # remove explicit 'self' from super().__init__(self, ...) calls
        method_node = remove_explicit_self_from_super_init(method_node)

        # Create a new FunctionDef node to be injected later
        new_method_name = f"super_{base_method_name}"
        new_method_node = gast.FunctionDef(
            name=new_method_name,
            args=method_node.args,
            body=method_node.body,
            decorator_list=method_node.decorator_list,
            returns=method_node.returns,
            type_comment=method_node.type_comment,
            type_params=method_node.type_params if hasattr(method_node, "type_params") else [],
        )
        new_method_node = gast.copy_location(new_method_node, method_node)
        self._method_nodes_to_add.append(new_method_node)

        # If we are about to inject a `super.__init__` call,
        # we need to make sure we correctly pass the current
        # class instance's `__init__` args/kwargs to the
        # `super.__init__` call to store them as `self._args`
        # and `self._kwargs`. This is needed for reconstructing an
        # object from its config during model cloning
        args, keywords = node.args, node.keywords
        if base_method_name == "__init__":
            args, keywords = self._get_init_arguments()

        # Update call to the super method with the new custom method
        call_node = gast.Call(
            func=gast.Attribute(
                value=gast.Name(
                    id="self",
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                ),
                attr=new_method_name,
                ctx=gast.Load(),
            ),
            args=args,
            keywords=keywords,
        )
        return gast.copy_location(call_node, node)

    def _get_init_arguments(self):
        args, args_list = [], []
        kwargs, kwargs_list = [], []

        # Retrieve the args node of the init method of the current class
        tree = gast.parse(self.transformer.object_like.source_code)
        args_node = [
            node
            for node in tree.body[0].body
            if getattr(node, "name", "") == "__init__"
        ][0].args

        # Func def had no kwonlyargs while keywords are defined with some defaults
        if args_node.defaults:
            args.extend(
                args_node.args[: -len(args_node.defaults)] + args_node.posonlyargs
            )
            kwargs.extend(
                args_node.kwonlyargs + args_node.args[-len(args_node.defaults) :]
            )
        else:
            args.extend(args_node.args + args_node.posonlyargs)
            kwargs.extend(args_node.kwonlyargs)

        # Iterate over the arguments
        for arg in args:
            if ast_to_source_code(arg).strip() == "self":
                continue

            if isinstance(arg, gast.Name):
                args_list.append(arg)
            else:
                arg_node = gast.Name(
                    id=arg.arg, ctx=gast.Load(), annotation=None, type_comment=None
                )

                args_list.append(arg_node)

        # Iterate over the keyword arguments
        for kwarg in kwargs:
            if isinstance(kwarg, gast.Name):
                kwarg_node = gast.keyword(arg=kwarg.id, value=kwarg)
            else:
                value_node = gast.Name(
                    id=kwarg.arg, ctx=gast.Load(), annotation=None, type_comment=None
                )
                kwarg_node = gast.keyword(arg=kwarg.arg, value=value_node)

            kwargs_list.append(kwarg_node)

        # Also pass in the `v`, `buffers` and `module_dict` in as a kwarg
        # We do this because we don't want the super init call to overwrite
        # these values and rather pass them on in the constructor itself
        for kwarg in ("v", "buffers", "module_dict"):
            value_node = (
                gast.parse(f"""getattr(self, "_{kwarg}", None)""").body[0].value
            )
            kwarg_node = gast.keyword(arg=kwarg, value=value_node)
            kwargs_list.append(kwarg_node)

        return args_list, kwargs_list
