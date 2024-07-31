import tensorflow

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dev
from .tensorflow__helpers import tensorflow_default_device_bknd
from .tensorflow__helpers import tensorflow_stack_bknd_


def tensorflow_dev(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    /,
    *,
    as_native: bool = False,
):
    if "keras.src.backend.tensorflow.core.Variable" in str(x.__class__):
        x = x.value
    if isinstance(x, tensorflow.TensorArray):
        x = tensorflow_stack_bknd_(x)
    dv = x.device
    if as_native:
        return dv
    dv = dv if dv else tensorflow_default_device_bknd(as_native=False)
    return tensorflow_as_ivy_dev(dv)
