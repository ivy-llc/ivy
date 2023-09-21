# import torch
from ivy_tests.test_ivy.test_frontends import NativeClass


torch_classes_to_ivy_classes = {}


def convtorch(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for torch."""
    if isinstance(argument, NativeClass):
        return torch_classes_to_ivy_classes.get(argument._native_class)
    return argument
