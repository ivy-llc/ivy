import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_linear(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    weight: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    bias: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    result = (
        tensorflow.matmul(x, weight, transpose_b=True)
        if len(x.shape) == len(weight.shape) == 2 and x.shape[-1] == weight.shape[-1]
        else tensorflow.einsum("...i,...ji->...j", x, weight)
    )
    if bias is not None:
        return tensorflow.add(result, bias)
    return result
