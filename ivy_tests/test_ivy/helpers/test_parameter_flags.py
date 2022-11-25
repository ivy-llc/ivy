from dataclasses import dataclass
from enum import Enum


class ContainerFlags:
    pass


class TestObjectType(Enum):
    FUNCTION = 1
    METHOD = 2
    INIT = 3


@dataclass(frozen=True)
class NumPositionalArg:
    object_type: TestObjectType = TestObjectType.FUNCTION


class NativeArrayFlags:
    pass


class AsVariableFlags:
    pass
