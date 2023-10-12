# import tensorflow
from ivy_tests.test_ivy.test_frontends import NativeClass


tensorflow_classes_to_ivy_classes = {}


def convtensor(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for
    tensorflow."""
    if isinstance(argument, NativeClass):
        return tensorflow_classes_to_ivy_classes.get(argument._native_class)
    return argument
