from enum import Enum


class ContainerFlags:
    pass


class TestObjectType(Enum):
    FUNCTION = 1
    METHOD = 2
    INIT = 3


class NumPositionalArg:  # TODO for backward compatibility only
    pass


class NumPositionalArgMethod:
    pass


class NumPositionalArgFn:
    pass


class NativeArrayFlags:
    pass


class AsVariableFlags:
    pass
