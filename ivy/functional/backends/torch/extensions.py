from typing import Optional, Union, Tuple, Literal, List, Sequence
from numbers import Number
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_data_not_indices_values_and_shape,
    _is_coo_not_csr,
)
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ivy.functional.backends.torch.elementwise import _cast_for_unary_op
import torch
import math


def is_native_sparse_array(x):
    return x.layout in [torch.sparse_coo, torch.sparse_csr]


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None,
):
    if _is_data_not_indices_values_and_shape(
        data, coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        ivy.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data
    elif _is_coo_not_csr(
        coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        return torch.sparse_coo_tensor(
            indices=coo_indices, values=values, size=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return torch.sparse_csr_tensor(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            size=dense_shape,
        )


def native_sparse_array_to_indices_values_and_shape(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return x.indices(), x.values(), x.size()
    elif x.layout == torch.sparse_csr:
        return [x.crow_indices(), x.col_indices()], x.values(), x.size()
    raise ivy.exceptions.IvyException("not a sparse COO/CSR Tensor")


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def sinc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinc(x, out=out)


sinc.support_native_out = True


def flatten(
    x: torch.Tensor,
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


def vorbis_window(
    window_length: torch.tensor,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.tensor(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )


vorbis_window.support_native_out = False


def lcm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.abs(torch.lcm(x1, x2, out=out))


lcm.support_native_out = True


def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[torch.dtype] = None,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.hann_window(
        window_length,
        periodic=periodic,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=None,
    )


hann_window.support_native_out = False


# noinspection PyUnresolvedReferences
def max_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = torch.nn.functional.pad(
        x,
        [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
        value=float("-inf"),
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.max_pool2d(x, kernel, strides, 0)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


max_pool2d.unsupported_dtypes = ("bfloat16", "float16")


def pad(
    x: torch.Tensor,
    /,
    pad_width: Tuple[int],
    *,
    mode: Optional[Literal["constant", "reflect", "edge", "wrap"]] = "constant",
    stat_length: Optional[Union[torch.Tensor, int]] = None,
    constant_values: Optional[Number] = 0,
    end_values: Optional[Number] = 0,
    reflect_type: Optional[Literal["even", "odd"]] = "even",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape == ():
        x = x.unsqueeze(0)
    if isinstance(pad_width, torch.Tensor):
        pad_width = pad_width.detach().cpu().numpy().tolist()
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    if mode == "constant":
        return torch.nn.functional.pad(
            x,
            pad_width_flat,
            mode=mode,
            value=constant_values,
        )
    else:
        x = x.unsqueeze(dim=0)
        if mode == "edge":
            mode = "replicate"
        elif mode == "wrap":
            mode = "circular"
            x = x.unsqueeze(dim=0)
        return torch.nn.functional.pad(x, pad_width_flat, mode=mode).squeeze()


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    )


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


def median(
    input: torch.tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    if hasattr(axis, "__iter__"):
        for dim in axis:
            input = torch.median(
                input,
                dim=dim,
                keepdim=keepdims,
                out=out,
            )
        return input
    else:
        return torch.median(
            input,
            dim=axis,
            keepdim=keepdims,
            out=out,
        )


def flipud(
    m: torch.Tensor,
    /,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.flipud(m)


flipud.support_native_out = False


def fmod(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fmod(x1, x2, out=None)


fmod.support_native_out = True
fmod.unsupported_dtypes = ("bfloat16",)


def fmax(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fmax(x1, x2, out=None)


fmax.support_native_out = True
