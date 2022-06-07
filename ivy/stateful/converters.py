"""Converters from Native Modules to Ivy Modules"""

# local
from ivy.backend_handler import current_backend as _cur_backend


def to_ivy_module(
    native_module=None,
    native_module_class=None,
    args=None,
    kwargs=None,
    device=None,
    devices=None,
    inplace_update=False,
):
    """
    Convert an instance of a trainable module from a native framework into a
    trainable ivy.Module instance.

    Parameters
    ----------
    native_module
        The module in the native framework to convert, required if native_module_class
        is not given.
        Default is None.
    native_module_class
        The class of the native module, required if native_module is not given.
        Default is None.
    args
        Positional arguments to pass to the native module class. Default is None.
    kwargs
        Key-word arguments to pass to the native module class. Default is None.
    device
        The device on which to create module variables. Default is None.
    devices
        The devices on which to create module variables. Default is None.
    inplace_update
        For backends with dedicated variable classes, whether to update these inplace.
        Default is False.

    Returns
    -------
    ret
        The new trainable ivy.Module instance.

    """
    return _cur_backend().to_ivy_module(
        native_module,
        native_module_class,
        args,
        kwargs,
        device,
        devices,
        inplace_update,
    )
