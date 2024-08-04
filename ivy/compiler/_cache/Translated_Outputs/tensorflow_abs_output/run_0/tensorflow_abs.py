import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_abs(
    x: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if not tensorflow.is_tensor(x):
        x = tensorflow.convert_to_tensor(x)
    if any(("uint" in x.dtype.name, "bool" in x.dtype.name)):
        return x
    return tensorflow.abs(x)
