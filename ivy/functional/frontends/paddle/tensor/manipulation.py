# local
from ..manipulation import *  # noqa: F401
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes

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

def _fill_diagonal_tensor_impl(x, y, offset=0, dim1=0, dim2=1, inplace=False):
    inshape = x.shape
    assert dim1 < len(inshape) and dim1 >= -len(inshape), (
        'dim1 should between [-rank,rank) in fill_diagonal_tensor_')
    assert dim2 < len(inshape) and dim2 >= -len(inshape), (
        'dim2 should between [-rank,rank) in fill_diagonal_tensor_')
    assert len(inshape) >= 2, (
        'Tensor dims should >= 2 in fill_diagonal_tensor_')
    dim1 %= len(inshape)
    dim2 %= len(inshape)

    predshape = []
    for i in range(len(inshape)):
        if i != dim1 and i != dim2:
            predshape.append(inshape[i])
    diaglen = min(
        min(inshape[dim1], inshape[dim1] + offset),
        min(inshape[dim2], inshape[dim2] - offset))
    predshape.append(diaglen)
    assert tuple(predshape) == tuple(y.shape), (
        "the y shape should be {}".format(predshape))
    if len(y.shape) == 1:
        y = y.reshape([1, -1])

    # if inplace:
    #     return _C_ops.fill_diagonal_tensor_(x, y, 'dim1', dim1, 'dim2', dim2,
    #                                         'offset', offset)
    # return _C_ops.fill_diagonal_tensor(x, y, 'dim1', dim1, 'dim2', dim2,
    #                                    'offset', offset)
    ivy.inplace_update(x, y)
    return x
    


def fill_diagonal_tensor_(x, y, offset=0, dim1=0, dim2=1, name=None):
    """
    **Notes**:
        **This API is ONLY available in Dygraph mode**

    This function fill the source Tensor y into the x Tensor's diagonal inplace.

    Args:
        x(Tensor): ``x`` is the original Tensor
        y(Tensor): ``y`` is the Tensor to filled in x
        dim1(int,optional): first dimension with respect to which to fill diagonal. Default: 0.
        dim2(int,optional): second dimension with respect to which to fill diagonal. Default: 1.
        offset(int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
        name(str,optional): Name for the operation (optional, default is None)

    Returns:
        Tensor: Tensor with diagonal filled with y.

    Returns type:
        list: dtype is same as x Tensor

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.ones((4, 3)) * 2
            y = paddle.ones((3,))
            x.fill_diagonal_tensor_(y)
            print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

    """
    return _fill_diagonal_tensor_impl(
        x, y, offset=offset, dim1=dim1, dim2=dim2, inplace=True)