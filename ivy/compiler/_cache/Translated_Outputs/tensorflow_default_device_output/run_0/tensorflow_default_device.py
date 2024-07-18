import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dev
from .tensorflow__helpers import tensorflow_as_native_dev
from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_dev
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_is_array

default_device_stack = []


def tensorflow_default_device(
    device: Optional[Union[str, str]] = None,
    /,
    *,
    item: Optional[Union[list, tuple, dict, tensorflow.Tensor, tf.Tensor]] = None,
    as_native: Optional[bool] = None,
):
    if tensorflow_exists(device):
        if as_native is True:
            return tensorflow_as_native_dev(device)
        elif as_native is False:
            return tensorflow_as_ivy_dev(device)
        return device
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_array(item):
            return tensorflow_dev(item, as_native=as_native)
    global default_device_stack
    if not default_device_stack:
        ret = "cpu"
    else:
        ret = default_device_stack[-1]
    if as_native:
        return tensorflow_as_native_dev(ret)
    return tensorflow_as_ivy_dev(ret)
