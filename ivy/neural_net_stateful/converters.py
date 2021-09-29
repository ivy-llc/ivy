"""
Converters from Native Modules to Ivy Modules
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def to_ivy_module(native_module=None, native_module_class=None, args=None, kwargs=None, f=None):
    """
    Convert an instance of a trainable module from a native framework into a trainable ivy.Module instance.

    :param native_module: The module in the native framework to convert, required if native_module_class is not given.
                          Default is None.
    :type native_module: native module instance, optional
    :param native_module_class: The class of the native module, required if native_module is not given. Default is None.
    :type native_module_class: class, optional
    :param args: Positional arguments to pass to the native module class. Default is None.
    :type args: list of any
    :param kwargs: Key-word arguments to pass to the native module class. Default is None.
    :type kwargs: dict of any
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The new trainable ivy.Module instance.
    """
    return _cur_framework(f=f).to_ivy_module(native_module, native_module_class, args, kwargs)
