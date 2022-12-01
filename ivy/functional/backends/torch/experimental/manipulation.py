from typing import Optional, Union, Sequence, Tuple, NamedTuple, List
from numbers import Number
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version
import torch
import ivy


def moveaxis(
    a: torch.Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
    x1: torch.tensor,
    x2: torch.tensor,
    /,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.heaviside(
        x1,
        x2,
        out=out,
    )


heaviside.support_native_out = True


def flipud(
    m: torch.Tensor,
    /,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.flipud(m)


flipud.support_native_out = False


def vstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.vstack(arrays, out=None)


def hstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.hstack(arrays, out=None)


def rot90(
    m: torch.Tensor,
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.rot90(m, k, axes)


def top_k(
    x: torch.Tensor,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_res = NamedTuple(
        "top_k", [("values", torch.Tensor), ("indices", torch.Tensor)]
    )
    if not largest:
        indices = torch.argsort(x, dim=axis)
        indices = torch.gather(indices, axis, torch.arange(k).expand_as(indices))
    else:
        x = -x
        indices = torch.argsort(x, dim=axis)
        indices = torch.gather(indices, axis, torch.arange(k).expand_as(indices))
        x = -x
    val = torch.gather(x, axis, indices)
    return topk_res(val, indices)


def fliplr(
    m: torch.Tensor,
    /,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.fliplr(m)


fliplr.support_native_out = False


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def i0(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.i0(x, out=out)


i0.support_native_out = True


def flatten(
    x: torch.Tensor,
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ivy.assertions.check_elem_in_list(order, ["C", "F"])
    if order == "F":
        return ivy.functional.experimental.flatten(
            x, start_dim=start_dim, end_dim=end_dim, order=order
        )
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


def vsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Tuple[int]],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.vsplit(ary, indices_or_sections)


def dsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Tuple[int]],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.dsplit(ary, indices_or_sections)


def atleast_1d(*arys: torch.Tensor) -> List[torch.Tensor]:
    transformed = torch.atleast_1d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


def dstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.dstack(arrays, out=None)


def atleast_2d(*arys: torch.Tensor) -> List[torch.Tensor]:
    transformed = torch.atleast_2d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


def atleast_3d(*arys: Union[torch.Tensor, bool, Number]) -> List[torch.Tensor]:
    transformed = torch.atleast_3d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
def take_along_axis(
    arr: torch.Tensor,
    indices: torch.Tensor,
    axis: int,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if arr.shape != indices.shape:
        raise ivy.exceptions.IvyException(
            "arr and indices must have the same shape;"
            + f" got {arr.shape} vs {indices.shape}"
        )
    indices = indices.long()
    return torch.take_along_dim(arr, indices, axis, out=out)


def hsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Tuple[int]],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.hsplit(ary, indices_or_sections)


take_along_axis.support_native_out = True
