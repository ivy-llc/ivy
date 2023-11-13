import numpy
from ivy_tests.test_ivy.test_frontends import NativeClass


numpy_classes_to_ivy_classes = {numpy._NoValue: None}


def convnumpy(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for numpy."""
    if isinstance(argument, NativeClass):
        return numpy_classes_to_ivy_classes.get(argument._native_class)
    return argument
