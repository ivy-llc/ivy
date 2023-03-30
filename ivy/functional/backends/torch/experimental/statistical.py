# global
from typing import Optional, Union, Tuple, Sequence
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def median(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return quantile(
        input,
        0.5,
        axis=axis,
        keepdims=keepdims,
        interpolation="midpoint",
    ).type_as(input)


median.support_native_out = False


def nanmean(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmean(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_unsupported_dtypes(
    {"1.11.0 and below": ("bfloat16", "bfloat32", "float16")}, backend_version
)
def quantile(
    a: torch.Tensor,
    q: Union[torch.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    temp = a.to(torch.float64)
    if isinstance(q, torch.Tensor):
        qt = q.to(torch.float64)
    else:
        qt = q
    if isinstance(axis, list) or isinstance(axis, tuple):
        dimension = len(a.size())
        for x in axis:
            axis1 = x
            for axis2 in range(x + 1, dimension):
                temp = torch.transpose(temp, axis1, axis2)
                axis1 = axis2
        temp = torch.flatten(temp, start_dim=dimension - len(axis))
        return torch.quantile(
            temp, qt, dim=-1, keepdim=keepdims, interpolation=interpolation, out=out
        )
    return torch.quantile(
        temp, qt, dim=axis, keepdim=keepdims, interpolation=interpolation, out=out
    )


quantile.support_native_out = True


def corrcoef(
    x: torch.Tensor,
    /,
    *,
    y: Optional[torch.Tensor] = None,
    rowvar: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = torch.concat([x, y], dim=axis)
        xarr = xarr.T if not rowvar else xarr

    return torch.corrcoef(xarr)


def nanmedian(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def bincount(
    x: torch.Tensor,
    /,
    *,
    weights: Optional[torch.Tensor] = None,
    minlength: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        ret = torch.bincount(x, minlength=minlength)
        ret = ret.to(x.dtype)
    else:
        ret = torch.bincount(x, weights=weights, minlength=minlength)
        ret = ret.to(weights.dtype)
    return ret


bincount.support_native_out = False


@with_unsupported_dtypes(
    {"1.11.0 and below": ("bfloat16", "bfloat32", "float16", "float32")}, 
    backend_version
)
def percentile(
    a: torch.Tensor,
    q: Union[Sequence[float], float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # if isinstance(axis, list) or isinstance(axis, tuple):
    #     temp = a.detach()
    #     dimension = len(a.size())
    #     for x in axis:
    #         axis1 = x
    #         for axis2 in range(x + 1, dimension):
    #             temp = torch.transpose(temp, axis1, axis2)
    #             axis1 = axis2
    #     temp = torch.flatten(temp, start_dim=dimension - len(axis))
    #     ret = torch.quantile(
    #         torch.tensor(temp), 
    #         torch.tensor(q / 100),
    #         dim=-1,
    #         keepdim=keepdims, 
    #         interpolation=interpolation,
    #         out=out
    #     )

    # ret = torch.quantile(
    #     torch.tensor(a), 
    #     torch.tensor(q / 100),
    #     dim=axis, 
    #     keepdim=keepdims, 
    #     interpolation=interpolation, 
    #     out=out
    # )

    # return ret
    temp = a
    qt = q / 100
    # if isinstance(q, torch.tensor):
    # qt = q.to(torch.float64)
    # else:
    # qt = q
    if isinstance(axis, list) or isinstance(axis, tuple):
        dimension = len(a.size())
        for x in axis:
            axis1 = x
            for axis2 in range(x + 1, dimension):
                temp = torch.transpose(temp, axis1, axis2)
                axis1 = axis2
        temp = torch.flatten(temp, start_dim=dimension - len(axis))

        print("\n\ninput a: ", a)
        print("\n\ninput q: ", q)
        ret = torch.quantile(
            temp, qt, dim=-1, keepdim=keepdims, interpolation=interpolation, out=out
        )
        print("\n\nret: ", ret)
        return ret
    print("\n\ninput a: ", a)
    print("\n\ninput q: ", q)
    ret = torch.quantile(
        temp, qt, dim=axis, keepdim=keepdims, interpolation=interpolation, out=out
    )
    print("\n\nret: ", ret)
    return ret


percentile.support_native_out = True
