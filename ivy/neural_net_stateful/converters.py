"""
Converters from Native Modules to Ivy Modules
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def to_ivy_module(native_module, f=None):
    """
    Convert an instance of a trainable module from a native framework into a trainable ivy.Module instance.

    :param native_module: The module in the native framework to convert.
    :type native_module: native module instance
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The new trainable ivy.Module instance.
    """
    return _cur_framework(f=f).to_ivy_module(native_module)
