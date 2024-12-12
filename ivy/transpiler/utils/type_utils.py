from __future__ import annotations
import enum
from types import FunctionType, MethodType, ModuleType
from typing import Union


class Types(enum.Enum):
    """
    Enum to hold information about
    live objects which can then be
    serialized/deserialized instead of
    caching entire live objects.

    # Note: Do not change the order of the enums
    nor the numbers infront of them
    """

    NoneType = 0
    MethodType = 1
    FunctionType = 2
    ClassType = 3
    ModuleType = 4

    @classmethod
    def get_type(
        cls,
        obj: Union[MethodType, FunctionType, type, ModuleType],
    ) -> Types:
        if isinstance(obj, MethodType):
            return Types.MethodType
        elif isinstance(obj, FunctionType):
            return Types.FunctionType
        elif isinstance(obj, type):
            return Types.ClassType
        elif isinstance(obj, ModuleType):
            return Types.ModuleType
