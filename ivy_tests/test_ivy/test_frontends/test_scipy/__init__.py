# import tensorflow
from ivy_tests.test_ivy.test_frontends import NativeClass


scipy_classes_to_ivy_classes = {}


def convscipy(argument):
    """Convert NativeClass in argument to ivy frontend counter part for tensorflow"""
    if isinstance(argument, NativeClass):
        return scipy_classes_to_ivy_classes.get(argument._native_class)
    return argument
