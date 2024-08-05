import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow__same_device
from .tensorflow__helpers import tensorflow_as_native_dev
from .tensorflow__helpers import tensorflow_dev
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_device(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if device is None:
        return x
    device = tensorflow_as_native_dev(device)
    current_dev = tensorflow_dev(x)
    if not tensorflow__same_device(current_dev, device):
        with tensorflow.device(f"/{device.upper()}"):
            return tensorflow.identity(x)
    return x
