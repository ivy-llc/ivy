# global
torch_scatter = None
import torch
from typing import Tuple, Union, Optional

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    if not keepdims and not axis and axis != 0:
        return torch.amax(input=x, out=out)
    return torch.amax(input=x, dim=axis, keepdim=keepdims, out=out)


def mean(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    return torch.mean(x, dim=axis, keepdim=keepdims, out=out)


def min(
    x: torch.Tensor,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    if not keepdims and not axis and axis != 0:
        return torch.amin(input=x, out=out)
    return torch.amin(input=x, dim=axis, keepdim=keepdims, out=out)


def prod(
    x: torch.Tensor,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: torch.dtype = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calculates the product of input array x elements.

    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which products must be computed. By default, the product must
        be computed over the entire array. If a tuple of integers, products must be
        computed over multiple axes. Default: None.
    keepdims
        bool, if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: False.
    dtype
        data type of the returned array. If None,
        if the default data type corresponding to the data type “kind” (integer or
        floating-point) of x has a smaller range of values than the data type of x
        (e.g., x has data type int64 and the default data type is int32, or x has data
        type uint64 and the default data type is int64), the returned array must have
        the same data type as x. if x has a floating-point data type, the returned array
        must have the default floating-point data type. if x has a signed integer data
        type (e.g., int16), the returned array must have the default integer data type.
        if x has an unsigned integer data type (e.g., uint16), the returned array must
        have an unsigned integer data type having the same number of bits as the default
        integer data type (e.g., if the default integer data type is int32, the returned
        array must have a uint32 data type). If the data type (either specified or
        resolved) differs from the data type of x, the input array should be cast to the
        specified data type before computing the product. Default: None.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array,  if the product was computed over the entire array, a zero-dimensional
        array containing the product; otherwise, a non-zero-dimensional array containing
        the products. The returned array must have a data type as described by the dtype
        parameter above.

    >>> x = torch.tensor([1, 2, 3])
    >>> z = torch.prod(x)
    >>> print(z)
    ivy.array(6)

    >>> x = torch.tensor([1, 0, 3])
    >>> z = torch.prod(x)
    >>> print(z)
    ivy.array(0)

    """
    if dtype is None:
        if x.dtype in [torch.int8, torch.int16]:
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int64, torch.int32]:
            dtype = torch.int64
        elif x.dtype == torch.bfloat16:
            dtype = torch.float16

    dtype = ivy.as_native_dtype(dtype)

    if axis is None:
        axis = x.dim() - 1
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = x.dim() - 1
        else:
            return torch.prod(
                torch.Tensor(
                    [
                        torch.prod(input=x, dim=i, dtype=dtype, keepdim=keepdims)
                        for i in axis
                    ]
                ),
                dtype=dtype,
                out=out,
            )
    return torch.prod(input=x, dim=axis, dtype=dtype, keepdim=keepdims, out=out)


def std(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.std(x, dim=axis, keepdim=keepdims, unbiased=False, out=out)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        if i == len(axis) - 1:
            x = torch.std(
                x,
                dim=a if keepdims else a - i,
                keepdim=keepdims,
                unbiased=False,
                out=out,
            )
        else:
            x = torch.std(
                x,
                dim=a if keepdims else a - i,
                keepdim=keepdims,
                unbiased=False,
                out=out,
            )
    return x


def sum(
    x: torch.Tensor,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: torch.dtype = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dtype is None:
        if x.dtype in [torch.int8, torch.int16]:
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int32, torch.int64]:
            dtype = torch.int64

    dtype = ivy.as_native_dtype(dtype)

    if axis is None:
        if out:
            return torch.sum(input=x, dtype=dtype, out=out)
        else:
            return torch.sum(input=x, dtype=dtype)
    elif type(axis) == list:
        return torch.sum(input=x, dim=axis, out=out)
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = 0
        else:
            return torch.sum(
                torch.Tensor(
                    [
                        torch.sum(input=x, dim=i, dtype=dtype, keepdim=keepdims)
                        for i in axis
                    ]
                ),
                dtype=dtype,
                out=out,
            )
    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims, out=out)


def var(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.var(x, dim=axis, keepdim=keepdims, unbiased=False, out=out)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        if i == len(axis) - 1:
            x = torch.var(
                x,
                dim=a if keepdims else a - i,
                keepdim=keepdims,
                unbiased=False,
                out=out,
            )
        else:
            x = torch.var(
                x, dim=a if keepdims else a - i, keepdim=keepdims, unbiased=False
            )
    return x


# Extra #
# ------#


def einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    return torch.einsum(equation, *operands)
