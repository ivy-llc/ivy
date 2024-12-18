"""Main file to hold DTOs to represent globals captured during transformations."""

# global
from __future__ import annotations
from typing import Dict
import gast
from enum import Enum

# local
from ...translations.data.object_like import BaseObjectLike
from ...utils.ast_utils import (
    ast_to_source_code,
    TranslatedContext,
)


class Position(Enum):
    """
    Represents the position of a global statement with respect to the object its bound to.
    """

    TOP = 0  # global statement above the func/class
    BOTTOM = 1  # global statement below the func/classs


class StackObjectLike:
    """
    Represents an object in the global stack during the transformation process.

    This class is used to keep track of global objects that are being processed,
    helping to avoid cyclic dependencies and manage inlining decisions.
    """

    def __init__(
        self,
        value_node: gast.Expr,
        target_str: str,
        object_like: BaseObjectLike,
    ):
        """
        Initialize a StackObjectLike instance.

        Args:
            node (gast.AST): The AST node representing the value_node in the global assignment.
            target_str (str): The string representation of the global assignment's target.
            object_like (Any): The object-like representation of the global.
        """
        self.value_node: gast.Expr = value_node
        self.target_str: str = target_str
        self.object_like: BaseObjectLike = object_like

    def __repr__(self) -> str:
        return f"StackObj(target_str='{self.target_str}', object_like={self.object_like.name})"


class GlobalObjectLike:
    """
    Represents a global object captured during the transformation process.

    This class holds information about various aspects of a global object, such
    as its type, position, filename, whether it has dependencies, and more.
    """

    def __init__(
        self,
        object_like: BaseObjectLike,
        *,
        target_node: gast.Name,
        value_node: gast.Expr,
        global_filename: str = "",
        global_dependencies: Dict = None,
        ctx: TranslatedContext = None,
    ):
        self.object_like: BaseObjectLike = object_like
        self.is_ivy_global = False  # whether the global stmt represents an ivy global (eg: X = ivy.mean, X= tensorflow_prod etc.)
        self.assignment_target = (
            ast_to_source_code(target_node).strip() if target_node else ""
        )  # the target of the assignment (eg: 'X' in X = y etc.)
        self.assignment_value = (
            value_node.value
            if isinstance(value_node, gast.Constant)
            and isinstance(value_node.value, str)
            else ast_to_source_code(value_node).strip() if value_node else ""
        )  # the value of the assignment (eg: 'y' in X = y etc.)
        value = value_node if value_node else gast.Constant(value="", kind=None)
        self.assignment_str = ast_to_source_code(
            gast.Assign(targets=[target_node], value=value)
        ).strip()  # the complete assignment str of the assignment (eg: 'X=y' in X = y etc.)
        self.origin = (
            self.object_like
        )  # the original object which this global belongs to. This comes in
        # practice in the case of nested globals, where the origin is the root global in the chain.
        # Initially, this defaults to the global's immediate parent object.
        self.position = Position.TOP  # the default position of the global.
        self.global_filename = global_filename  # the name of the file within which this global is to be emitted
        self.global_dependencies = (
            global_dependencies  # dependencies on the RHS of the global assignment.
        )
        self.ctx = ctx  # the context in which this global is being used

    def __eq__(self, other):
        if isinstance(other, GlobalObjectLike):
            return self.assignment_target == other.assignment_target
        return False

    def __hash__(self):
        return hash(self.assignment_target)

    def __repr__(self) -> str:
        return f"GlobalObjectLike({self.assignment_str})"

    def to_dict(self: GlobalObjectLike) -> Dict:
        return {
            "object_like": self.object_like.to_dict(),
            "is_ivy_global": self.is_ivy_global,
            "origin": self.origin.to_dict(),
            "position": self.position.name,
            "global_filename": self.global_filename,
            "global_dependencies": self.global_dependencies,
            "assignment_target": self.assignment_target,
            "assignment_value": self.assignment_value,
            "assignment_str": self.assignment_str,
        }

    @staticmethod
    def from_dict(data: Dict) -> GlobalObjectLike:
        object_like = BaseObjectLike.from_dict(data["object_like"])
        glob_origin = BaseObjectLike.from_dict(data["origin"])

        # Reconstruct the target_node and value_node from the assignment strings
        target_node = (
            gast.parse(data["assignment_target"]).body[0].value
            if data["assignment_target"]
            else None
        )
        value_node = (
            gast.parse(data["assignment_value"]).body[0].value
            if data["assignment_value"]
            else None
        )

        global_obj = GlobalObjectLike(
            object_like=object_like,
            target_node=target_node,
            value_node=value_node,
        )
        global_obj.assignment_str = data["assignment_str"]
        global_obj.is_ivy_global = data["is_ivy_global"]
        global_obj.origin = glob_origin
        global_obj.position = Position[data["position"]]
        global_obj.global_filename = data["global_filename"]
        global_obj.global_dependencies = data["global_dependencies"]
        return global_obj
