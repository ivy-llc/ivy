import tensorflow

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dev
from .tensorflow__helpers import tensorflow_default_device
from .tensorflow__helpers import tensorflow_stack_1


def tensorflow_dev(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    /,
    *,
    as_native: bool = False,
):
    if isinstance(x, tensorflow.TensorArray):
        x = tensorflow_stack_1(x)
    dv = x.device
    if as_native:
        return dv
    dv = dv if dv else tensorflow_default_device(as_native=False)
    return tensorflow_as_ivy_dev(dv)
