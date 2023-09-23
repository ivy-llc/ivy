# import tensorflow
from ivy_tests.test_ivy.test_frontends import NativeClass


onnx_classes_to_ivy_classes = {}


def convtensor(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for onnx."""
    if isinstance(argument, NativeClass):
        return onnx_classes_to_ivy_classes.get(argument._native_class)
    return argument
