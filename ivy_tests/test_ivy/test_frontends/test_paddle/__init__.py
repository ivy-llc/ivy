# import paddle
from ivy_tests.test_ivy.test_frontends import NativeClass


paddle_classes_to_ivy_classes = {}


def convpaddle(argument):
    """Convert NativeClass in argument to ivy frontend counter part for paddle."""
    if isinstance(argument, NativeClass):
        return paddle_classes_to_ivy_classes.get(argument._native_class)
    return argument
