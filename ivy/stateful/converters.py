"""
Converters from Native Modules to Ivy Modules
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def to_ivy_module(native_module=None, native_module_class=None, args=None, kwargs=None, dev=None, devs=None,
                  inplace_update=False):
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
    :param dev: The device on which to create module variables. Default is None.
    :type dev: ivy.Device, optional
    :param devs: The devices on which to create module variables. Default is None.
    :type devs: sequence of str, optional
    :param inplace_update: For backends with dedicated variable classes, whether to update these inplace.
                           Default is False.
    :type inplace_update: bool, optional
    :return: The new trainable ivy.Module instance.
    """
    return _cur_framework().to_ivy_module(native_module, native_module_class, args, kwargs, dev, devs, inplace_update)
