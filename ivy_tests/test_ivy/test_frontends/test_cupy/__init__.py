import cupy
from ivy_tests.test_ivy.test_frontends import NativeClass


cupy_classes_to_ivy_classes = {cupy._numpy._NoValue: None}


def convcupy(argument):
    """Convert NativeClass in argument to ivy frontend counter part for cupy.numpy interface."""
    if isinstance(argument, NativeClass):
        return cupy_classes_to_ivy_classes.get(argument._native_class)
    return argument
