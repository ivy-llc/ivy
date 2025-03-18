# global
import copy
import gast
import ivy

# local
from ..base_transformer import (
    BaseTransformer,
)
from ...transformer import Transformer
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import ast_to_source_code
from ....utils.api_utils import (
    get_function_from_modules,
    is_backend_api,
)

from ... import transformer_globals as glob
from ....utils.type_utils import Types
from ....utils.naming_utils import NAME_GENERATOR


class NameNodeVisitor(gast.NodeVisitor):
    def __init__(self):
        self.has_name_node = False

    def visit_Name(self, node):
        self.has_name_node = True


class BaseDundersTransformer(BaseTransformer):
    """
    A class to Transforms implicit `__getitem__` and `__setitem__` calls to explicit function
    calls and in-place operations to explicit assignments in an AST.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

        self.methods_not_to_ignore = ("__init__", "forward")
        self.classes_to_ignore = ("ModuleList", "ModuleDict", "Sequential")
        import ivy.functional.frontends.torch

        self.stateful_classes = [
            ivy.Module,
            ivy.NativeModule,
            ivy.functional.frontends.torch.nn.Module,
        ]

    def transform(self):
        self.visit(self.root)

    def visit_Name(self, node):
        return node

    def visit_ClassDef(self, node):
        # no need to handle subscripts for stateful container classes
        if any(
            node.name[len(NAME_GENERATOR.old_prefix) :] == class_name
            for class_name in self.classes_to_ignore
        ):
            return node

        for stmt in node.body:
            self.visit(stmt)
        return node

    def visit_FunctionDef(self, node):
        # no need to handle subscripts for internal stateful methods
        # when translating classes (eg:_create_variables, named_parameters)
        if (
            self.transformer.object_like.type == Types.ClassType
            and node.name not in self.methods_not_to_ignore
            and any(node.name in dir(cls) for cls in self.stateful_classes)
        ):
            return node

        # no need to handle subscripts for backend ivy functions
        if self.transformer.object_like.is_backend_api:
            return node

        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node):
        # Transform augmented operations to explicit assignments
        node = gast.Assign(
            targets=[node.target],
            value=gast.BinOp(left=node.target, op=node.op, right=node.value),
        )
        node.value = copy.deepcopy(node.value)
        self.set_context_to_load(node.value)
        node = self.visit_Assign(node)
        return node

    def set_context_to_load(self, node):
        # Recursively set context of all nodes to Load
        for child in gast.iter_child_nodes(node):
            if hasattr(child, "ctx"):
                child.ctx = gast.Load()
            else:
                self.set_context_to_load(child)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], gast.Subscript):
            subscript_node = node.targets[0]

            # If we are simply assigning to a `self.__dict__` here
            # we avoid transforming it since it can lead to certain
            # recursion issues. See: kornia.core.tensor_wrapper.TensorWrapper
            # ref: https://github.com/kornia/kornia/blob/f6ff780d17f21bff1b3b26472ea19dfdc95cf899/kornia/core/tensor_wrapper.py#L30
            is___dict___assignment = (
                isinstance(subscript_node.value, gast.Attribute)
                and ast_to_source_code(subscript_node.value).strip() == "self.__dict__"
            )

            if not is___dict___assignment and self.should_transform(subscript_node):
                slice_args = self.convert_slices(subscript_node.slice)
                value = self.visit(node.value)
                new_node = self.transform_setitem(subscript_node, slice_args, value)

                return new_node

        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)

        if self.should_transform(node):
            slice_args = self.convert_slices(node.slice)

            if isinstance(node.ctx, gast.Load):
                new_node = self.transform_getitem(node, slice_args)
                return new_node

        return node

    def should_transform(self, node: gast.Subscript):
        # DO NOT transform type annotations which also show up as gast.Subcript nodes.
        value = ast_to_source_code(node.value).strip()
        func_obj = get_function_from_modules(value, self.transformer.object_module)
        if hasattr(func_obj, "__module__") and func_obj.__module__ == "typing":
            return False

        # only transform if dunder:
        # 1) is __setitem__ call
        # 2) the query is complex (i.e contains a gast.Name node)
        if isinstance(node.ctx, gast.Store):
            return True

        visitor = NameNodeVisitor()
        visitor.visit(node.slice)

        return visitor.has_name_node

    def convert_slices(self, slice_node):
        get_slice_val = lambda val: val if val else gast.parse("None").body[0].value
        if isinstance(slice_node, gast.Slice):
            return gast.Call(
                func=gast.Name(
                    id="slice", ctx=gast.Load(), type_comment=None, annotation=None
                ),
                args=[
                    get_slice_val(slice_node.lower),
                    get_slice_val(slice_node.upper),
                    get_slice_val(slice_node.step),
                ],
                keywords=[],
            )
        elif isinstance(slice_node, gast.Tuple):
            return gast.Tuple(
                elts=[self.convert_slices(elt) for elt in slice_node.elts],
                ctx=gast.Load(),
            )
        else:
            return slice_node

    def transform_getitem(self, node, slice_args):
        glob.CONFLICTING_METHODS.add("get_item")
        return gast.Call(
            func=gast.Attribute(
                value=gast.parse(f"ivy").body[0].value,
                attr="get_item",
                ctx=gast.Load(),
            ),
            args=[node.value, slice_args],
            keywords=[],
        )

    def transform_setitem(self, node, slice_args, value):
        glob.CONFLICTING_METHODS.add("set_item")
        return gast.Assign(
            targets=[node.value],
            value=gast.Call(
                func=gast.Attribute(
                    value=gast.parse(f"ivy").body[0].value,
                    attr="set_item",
                    ctx=gast.Load(),
                ),
                args=[node.value, slice_args, value],
                keywords=[],
            ),
        )
