"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and
signature."""

# global
import torch
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, torch.Tensor] = 0.0,
    high: Union[float, torch.Tensor, None] = 1.0,
    shape: Optional[Union[torch.Tensor, ivy.NativeShape, Sequence[int]]] = None,
    dtype: torch.dtype,
    device: torch.device = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if high is None:
        # default to float32, as this is the tf standard
        high = (
            torch.finfo(dtype).max
            if dtype is not None
            else torch.finfo(torch.float32).max
        )
    rand_range = high - low
    if seed:
        torch.manual_seed(seed)
    if torch.is_tensor(shape):
        shape = shape.tolist()
    return (
        torch.rand(shape, device=device, dtype=torch.float) * rand_range + low
    ).type(dtype)


def random_normal(
    *,
    mean: Union[float, torch.Tensor] = 0.0,
    std: Union[float, torch.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape
    dtype = ivy.as_native_dtype(dtype)
    if seed:
        torch.manual_seed(seed)
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        return torch.normal(mean, std, shape, out=out).type(dtype).to(device)
    return torch.normal(mean, std, out=out).type(dtype).to(device)


random_normal.support_native_out = True


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, backend_version)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[torch.Tensor] = None,
    replace: bool = True,
    device: torch.device = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if probs is None:
        probs = (
            torch.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    if seed:
        torch.manual_seed(seed)
    return torch.multinomial(probs.float(), num_samples, replace, out=out).to(device)


multinomial.support_native_out = True


def randint(
    low: Union[int, torch.Tensor],
    high: Union[int, torch.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: torch.device = None,
    dtype: Optional[Union[torch.dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    rand_range = high - low
    if seed:
        torch.manual_seed(seed)
    return (torch.rand(shape, device=device) * rand_range + low).to(dtype)


def seed(*, seed_value: int = 0):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            from torch import mps

            mps.manual_seed(seed_value)
    return


def shuffle(
    x: torch.Tensor,
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x.shape) == 0:
        return x
    batch_size = x.shape[0]
    if seed:
        torch.manual_seed(seed)
    return torch.index_select(x, 0, torch.randperm(batch_size), out=out)


shuffle.support_native_out = True
