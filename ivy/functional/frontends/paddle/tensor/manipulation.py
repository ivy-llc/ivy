# local
from ..manipulation import *  # noqa: F401
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes


# Add the index_add_ function
@with_unsupported_dtypes(
    {"2.5.1 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_ivy_arrays_and_back
def index_add_(x, index, value):
    """
    Index_add for PaddlePaddle Frontend.

    Args:
        x (ivy.Array): The input array.
        index (ivy.Array): The indices where values should be added.
        value (ivy.Array): The values to add to the specified indices.

    Returns:
        ivy.Array: The modified input array with values added at specified indices.
    """
    # Implement the index_add_ function
    ret = index_add(x, index, value)
    ivy.inplace_update(x, ret)
    return x

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/manipulation.py`.


@with_unsupported_dtypes(
    {"2.5.1 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_ivy_arrays_and_back
def reshape_(x, shape):
    ret = ivy.reshape(x, shape)
    ivy.inplace_update(x, ret)
    return x
