# global
import gast
import ivy
import inspect
import os
import importlib

# local
from ....transformations import transformer_globals as glob
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import (
    ast_to_source_code,
)
from .base_transformer import (
    BaseCodePostProcessor,
)
from ....utils.type_utils import Types


class IvyCodePostProcessor(BaseCodePostProcessor):
    """
    Base class to perform post-processing on the final gast AST.
    This handles common transformations and can be extended for specific
    backends like TensorFlow, JAX etc.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        new_name="tensor",
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        self.new_name = new_name
        self._intermediate_var_name = "ag__result_list"
        self._list_comp_counter = 0
        self._lis_comp_blocklist = set(glob.LIST_COMP_TRANSFORMATION_BLOCKLIST).union(
            set(
                [
                    x
                    for y in (
                        ivy.functional.nest,
                        importlib.import_module(
                            "ivy.functional.frontends.torch"
                        ).nn.Module,
                    )
                    for x in dir(y)
                    if inspect.isfunction(getattr(y, x)) and not x.startswith("__")
                ]
            )
        )
        self._in_method = False
        self._in_try_block = False

    def transform(self):
        self.visit(self.root)

    def visit_Module(self, node):
        node = self._maybe_convert_list_comps_to_loops(node)
        self.generic_visit(node)
        return node

    def visit_Try(self, node):
        self._in_try_block = True
        self.generic_visit(node)
        self._in_try_block = False
        return node

    def visit_Import(self, node):
        return [] if not self._in_try_block else node

    def visit_ImportFrom(self, node):
        return [] if not self._in_try_block else node

    def visit_ClassDef(self, node):
        for i, base in enumerate(node.bases):
            node.bases[i] = self.visit(base)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        node = self._maybe_convert_list_comps_to_loops(node)
        node = self._maybe_rename_forward_to_call(node)
        node = self._maybe_add_tf_name_scope(node)
        node = self._maybe_modify_inplace_update_fn(node)
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

    def visit_keyword(self, node):
        self.generic_visit(node)

        if os.environ.get(
            "APPLY_TRANSPOSE_OPTIMIZATION", None
        ) == "true" and node.arg in ["data_format", "filter_format"]:
            if isinstance(node.value, gast.Constant):
                if node.value.value == "channel_first":
                    node.value.value = "channel_last"
                elif node.value.value == "NCHW":
                    node.value.value = "NHWC"
                elif node.value.value == "NCS":
                    node.value.value = "NSC"

        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if self.transformer.object_like.is_translated_api:
            if node.id == "arr":
                node.id = self.new_name
                return node

        if node.id in self.configuration.dtype_mapping[self.transformer.target]:
            return (
                gast.parse(
                    self.configuration.dtype_mapping[self.transformer.target][node.id]
                )
                .body[0]
                .value
            )

        elif node.id in self.configuration.hf_cls_mapping[self.transformer._target]:
            new_cls_name, from_import = self.configuration.hf_cls_mapping[
                self.transformer._target
            ][node.id]
            _, from_mod, _, from_obj = from_import.split(" ")
            self.transformer._from_imports.add((from_mod, from_obj, None))
            return gast.parse(new_cls_name).body[0].value

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_str = ast_to_source_code(node).strip()

        if attr_str == "ivy.Array":
            return self._handle_ivy_array(node)
        elif attr_str == "ivy.Variable":
            return self._handle_ivy_variable(node)
        elif attr_str == "ivy.Module":
            return self._handle_ivy_module(node)
        elif any(
            ivy_excp in attr_str
            for ivy_excp in ("ivy.utils.exceptions.", "ivy.exceptions.")
        ):
            return gast.Name(
                id="Exception", ctx=gast.Load(), type_comment=None, annotation=None
            )
        elif attr_str in self.configuration.dtype_mapping[self.transformer.target]:
            return (
                gast.parse(
                    self.configuration.dtype_mapping[self.transformer.target][attr_str]
                )
                .body[0]
                .value
            )
        elif attr_str in self.configuration.ivy_globs:
            return gast.parse(self.configuration.ivy_globs[attr_str]).body[0].value
        elif attr_str in self.configuration.ivy_cls_mapping[self.transformer._target]:
            return (
                gast.parse(
                    self.configuration.ivy_cls_mapping[self.transformer._target][
                        attr_str
                    ]
                )
                .body[0]
                .value
            )
        elif (
            attr_str in self.configuration.native_cls_mapping[self.transformer._target]
        ):
            return (
                gast.parse(
                    self.configuration.native_cls_mapping[self.transformer._target][
                        attr_str
                    ]
                )
                .body[0]
                .value
            )

        # TODO: remove this once stateful attribute conflicts (self.layers, self.training) has been resolved
        elif attr_str.startswith("self."):
            attr_name = attr_str.replace("self.", "")
            try:
                native_attr = getattr(ivy.NativeModule, attr_name)
            except AttributeError:
                native_attr = None
            # resolving name conflict with read-only properties of the stateful class
            if isinstance(native_attr, property) and attr_name not in (
                "training",
                "dtype",
                "device",
                "v",
                "buffers",
            ):
                new_attr = "self.pt_" + attr_name  # rename the attribute
                return gast.parse(new_attr).body[0].value

        if self.transformer.object_like.is_ivy_api and node.attr in (
            "data",
            "_data",
        ):
            # delete the '.data' attribute as we're no longer dealing with ivy arrays
            return node.value
        # TODO: add a more robust fix for this (eg: self.layers.conv2d.weight.data --> self.layers.conv2d.weight)
        if node.attr in ("data", "_data") and isinstance(node.value, gast.Attribute):
            node_parts = ast_to_source_code(node.value).strip().split(".")
            if node_parts[0] == "self" and node_parts[-1] in ("weight", "bias"):
                return node.value

        if node.attr == "device":
            return self._maybe_convert_device_attribute(node)
        return node

    def visit_With(self, node):
        self.generic_visit(node)
        for item in node.items:
            if isinstance(item.context_expr, gast.Call):
                func = item.context_expr.func
                if ast_to_source_code(func).strip() in (
                    "ivy.ArrayMode",
                    "ivy.PreciseMode",
                ):
                    # Replace the With node with the statements in its body
                    return node.body
        return node

    def visit_Call(self, node):
        func_name = ast_to_source_code(node.func).strip()
        if func_name == "isinstance":
            # Check if the call is to isinstance and if so,
            # maybe transform the type check argument to
            # convert `isinstance(..., (ivy.Array, ivy.Array))` calls to
            # `isinstance(..., (tensorflow.Tensor, tensorflow.Variable))`
            node.args[1] = self._maybe_replace_ivy_array_type_check(node.args[1])

        if func_name in self.configuration.default_dtype_mapping.keys():
            as_native = self.get_as_native_value(node)
            backend = self.transformer._target
            new_value = self.get_dtype_value(func_name, backend, as_native)
            return gast.parse(new_value).body[0].value
        elif func_name == "ivy.current_backend_str":
            return gast.Constant(value=ivy.current_backend_str(), kind=None)
        elif (
            os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
            and "_ConvNd" in self.transformer.object_like.name
            and "empty_frnt" in func_name
        ):
            if isinstance(node.args[0], gast.Tuple):
                dims = node.args[0].elts
                # Check if the dimensions are in the PyTorch format and determine the pattern
                if (
                    len(dims) >= 3
                    and isinstance(dims[0], gast.Name)
                    and dims[0].id in ("in_channels", "out_channels")
                ):

                    def is_out_channels_binop(dim):
                        # Check if the dimension is out_channels // groups (BinOp)
                        return (
                            isinstance(dim, gast.BinOp)
                            and isinstance(dim.left, gast.Name)
                            and dim.left.id == "out_channels"
                        )

                    def is_in_channels_binop(dim):
                        # Check if the dimension is in_channels // groups (BinOp)
                        return (
                            isinstance(dim, gast.BinOp)
                            and isinstance(dim.left, gast.Name)
                            and dim.left.id == "in_channels"
                        )

                    # Pattern for transposed=True: (in_channels, out_channels // groups, *kernel_size)
                    if dims[0].id == "in_channels" and is_out_channels_binop(dims[1]):
                        in_channels = dims[0]
                        out_channels_groups = dims[1]
                        kernel_size = dims[2:]
                        if self.transformer.target == "tensorflow":
                            # (*kernel_size, out_channels // groups, in_channels)
                            node.args[0].elts = kernel_size + [
                                out_channels_groups,
                                in_channels,
                            ]
                        else:
                            # Flax Conv uses orders the weight for both Conv and ConvTranspose in the same way.
                            # (*kernel_size, in_channels, out_channels // groups)
                            node.args[0].elts = kernel_size + [
                                in_channels,
                                out_channels_groups,
                            ]
                    # Pattern for transposed=False: (out_channels, in_channels // groups, *kernel_size)
                    elif dims[0].id == "out_channels" and is_in_channels_binop(dims[1]):
                        out_channels = dims[0]
                        in_channels_groups = dims[1]
                        kernel_size = dims[2:]
                        # Transform to TensorFlow/JAX channel-last format
                        node.args[0].elts = kernel_size + [
                            in_channels_groups,
                            out_channels,
                        ]

        self.generic_visit(node)
        node = self._maybe_replace_with_native_array_calls(node)
        node = self._maybe_rename_forward_to_call(node)
        return node

    def visit_Assign(self, node):
        if self._in_method:
            self._handle_tf_name_scope(node)
        self.generic_visit(node)
        return node

    def _maybe_replace_ivy_array_type_check(self, arg: gast.Tuple):
        # If the argument is a tuple, check for the matching patterns and transform
        if isinstance(arg, gast.Tuple):
            transformed_elts = self._replace_ivy_array_pattern(arg.elts)
            return gast.Tuple(elts=transformed_elts, ctx=gast.Load())
        else:
            # Handle the single type case, but since we are looking for specific tuple patterns,
            # this case won't apply here.
            return arg

    def _maybe_rename_forward_to_call(self, node):
        is_class = self.transformer.object_like.type == Types.ClassType
        if isinstance(node, gast.Call):
            return node
        is_hf_pretrained_class = (
            is_class and self.transformer.object_like.is_hf_pretrained_class
        )
        is_forward_method = node.name == "forward"
        is_add_module_method = node.name == "add_module"
        if is_class and is_forward_method:
            if is_hf_pretrained_class:
                # rename `forward`
                node.name = self._get_forward_name(node)
            if (
                os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION", None) == "true"
                and "Sequential" in self.transformer.object_like.name
            ):
                # Transpose Optimization # 1:
                # modify the forward method.
                """
                for module in self --> "for i,module in enumerate(self)
                """
                # this change helps out in the `handle_transpose_in_input_and_output` decorator
                # used during the transpose optimization.
                self._maybe_modify_sequential_forward(node)
            if is_hf_pretrained_class and self.transformer.object_like.is_root_obj:
                self._maybe_add_unpack_inputs_decorator(node)

        if is_class and is_add_module_method:
            self._maybe_modify_add_module(node)
            self._maybe_modify_sequential_add_module(node)
        return node

    def _maybe_modify_sequential_forward(self, node):
        def transform_for_loop(node):
            enum_call = gast.Call(
                func=gast.Name(
                    id="enumerate", ctx=gast.Load(), annotation=None, type_comment=None
                ),
                args=[node.iter],
                keywords=[],
            )
            node.iter = enum_call
            index_var = gast.Name(
                id="i", ctx=gast.Store(), annotation=None, type_comment=None
            )
            node.target = gast.Tuple(elts=[index_var, node.target], ctx=gast.Store())

        for child in gast.iter_child_nodes(node):
            if isinstance(child, gast.For):
                transform_for_loop(child)
                break

    def _maybe_modify_sequential_add_module(self, node):
        # this change helps out when using native keras layers
        for child in gast.iter_child_nodes(node):
            if (
                isinstance(child, gast.If)
                and isinstance(child.test.values[0].operand, gast.Call)
                and child.test.values[0].operand.func.id == "isinstance"
            ):
                self._transform_isinstance_check(child.test.values[0].operand)
                break

    def _convert_list_comps(self, node: gast.ListComp):
        # Extract the parts of the list comprehension
        nodes = []
        if isinstance(node.elt, gast.ListComp):
            nodes += self._convert_list_comps(node.elt).body
            node.elt = (
                gast.parse(
                    f"{self._intermediate_var_name}_{self._list_comp_counter - 1}"
                )
                .body[0]
                .value
            )

        generator = node.generators[0]
        ifs = generator.ifs
        iter_expr = generator.iter
        targets = generator.target

        loop_body = [
            gast.Assign(
                targets=[
                    gast.Name(
                        id="res", ctx=gast.Store(), type_comment=None, annotation=None
                    )
                ],
                value=node.elt,
            ),
            gast.Expr(
                value=gast.Call(
                    func=gast.Attribute(
                        value=gast.Name(
                            id=f"{self._intermediate_var_name}_{self._list_comp_counter}",
                            ctx=gast.Load(),
                            annotation=None,
                            type_comment=None,
                        ),
                        attr="append",
                        ctx=gast.Load(),
                    ),
                    args=[
                        gast.Name(
                            id="res",
                            ctx=gast.Load(),
                            type_comment=None,
                            annotation=None,
                        )
                    ],
                    keywords=[],
                )
            ),
        ]

        # Deal with any if-else conditions in there
        # Case 1. Single if condition:
        if ifs:
            loop_body = nodes + [
                gast.If(
                    test=ifs[0],
                    body=loop_body,
                    orelse=[],
                )
            ]
        else:
            loop_body = nodes + loop_body

        # Create a for loop with append statements
        for_loop = gast.For(
            target=targets,
            iter=iter_expr,
            body=loop_body,
            orelse=[],
            type_comment=None,
        )

        # Create a new list to store the elements
        result_list = gast.List(elts=[], ctx=gast.Load())
        result_list_assign = gast.Assign(
            targets=[
                gast.Name(
                    id=f"{self._intermediate_var_name}_{self._list_comp_counter}",
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None,
                )
            ],
            value=result_list,
        )
        # Replace the list comprehension with the new code
        new_node = gast.Module(
            body=[result_list_assign, for_loop],
            type_ignores=[],
        )
        node = gast.fix_missing_locations(new_node)

        self._list_comp_counter += 1

        self.generic_visit(node)
        return node

    def _maybe_convert_list_comps_to_loops(self, node):
        if not any(
            x in getattr(node, "name", "").strip()
            or x in self.transformer.object_like.name
            for x in self._lis_comp_blocklist
        ):
            new_body = []
            for body_node in node.body:
                # Convert list comprehensions into explicit for loops
                if isinstance(body_node, gast.Assign):
                    if isinstance(body_node.value, gast.ListComp):
                        new_body += self._convert_list_comps(body_node.value).body
                        body_node.value = (
                            gast.parse(
                                f"{self._intermediate_var_name}_{self._list_comp_counter - 1}"
                            )
                            .body[0]
                            .value
                        )

                    elif (
                        isinstance(body_node.value, gast.Call)
                        and body_node.value.args
                        and isinstance(body_node.value.args[0], gast.ListComp)
                    ):
                        new_body += self._convert_list_comps(
                            body_node.value.args[0]
                        ).body
                        body_node.value.args[0] = (
                            gast.parse(
                                f"{self._intermediate_var_name}_{self._list_comp_counter - 1}"
                            )
                            .body[0]
                            .value
                        )

                body_node = gast.fix_missing_locations(body_node)
                new_body.append(body_node)

            node.body = new_body
            node = gast.fix_missing_locations(node)
        return node

    def get_as_native_value(self, node):
        as_native = "false"
        for kw in node.keywords:
            if kw.arg == "as_native":
                if isinstance(kw.value, gast.Constant) and isinstance(
                    kw.value.value, bool
                ):
                    as_native = str(kw.value.value).lower()
                else:
                    as_native = "false"
        return as_native

    def get_dtype_value(self, dtype_key, backend, as_native):
        dtype_mapping = self.configuration.default_dtype_mapping
        value = dtype_mapping[dtype_key][backend][as_native]
        if as_native == "false":
            value = f'"{value}"'
        return value

    def _handle_ivy_array(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _handle_ivy_variable(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _handle_ivy_module(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _handle_assign_transform(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _handle_tf_name_scope(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _transform_isinstance_check(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _get_forward_name(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _replace_ivy_array_pattern(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_add_unpack_inputs_decorator(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_add_tf_name_scope(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_modify_add_module(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_replace_with_native_array_calls(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_modify_inplace_update_fn(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()

    def _maybe_convert_device_attribute(self, node):
        """To be implemented by child classes for backend-specific handling."""
        raise NotImplementedError()
