import torch


torch_classes_to_ivy_classes = {}


def convtorch(argument):
    """Convert NativeClass in argument to ivy frontend counter part for torch"""
    if isinstance(argument, NativeClass):
        return torch_classes_to_ivy_classes.get(argument._native_class)
    return argument
