from .tensorflow__helpers import tensorflow__get_first_array
from .tensorflow__helpers import tensorflow_default_device


def tensorflow__get_preferred_device(args, kwargs):
    device = None
    if "device" in kwargs and kwargs["device"] is not None:
        return device
    if not False:
        arr_arg = tensorflow__get_first_array(*args, **kwargs)
        return tensorflow_default_device(item=arr_arg, as_native=True)
    return tensorflow_default_device(as_native=True)
