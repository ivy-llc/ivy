# import jax
from ivy_tests.test_ivy.test_frontends import NativeClass


jax_classes_to_ivy_classes = {}


def convjax(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for jax."""
    if isinstance(argument, NativeClass):
        return jax_classes_to_ivy_classes.get(argument._native_class)
    return argument
