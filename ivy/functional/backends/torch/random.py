"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import Optional, List, Union, Sequence

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.torch.data_type import as_native_dtype

# Extra #
# ------#


def random_uniform(
    low: Union[float, torch.Tensor] = 0.0,
    high: Union[float, torch.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    device: torch.device,
    dtype = torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if shape is None:
        shape = []
    rand_range = high - low
    return (
        torch.rand(shape, device=default_device(device), dtype=dtype, out=out)
        * rand_range
        + low
    )


random_uniform.support_native_out = True


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    mean = float(mean) if isinstance(mean, (torch.Tensor, ivy.Array)) else mean
    std = float(std) if isinstance(std, (torch.Tensor, ivy.Array)) else std
    return torch.normal(
        mean, std, true_shape, device=default_device(device),
        dtype=as_native_dtype(dtype), out=out
    )


random_normal.support_native_out = True


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[torch.Tensor] = None,
    replace: bool = True,
    *,
    device: torch.device,
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
    return torch.multinomial(probs.float(), num_samples, replace, out=out).to(
        default_device(device)
    )


multinomial.support_native_out = True


def randint(
    low: int,
    high: int,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.randint(low, high, shape, out=out, device=default_device(device))


randint.support_native_out = True


def seed(
    seed_value: int = 0,
) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size, out=out), out=out)


shuffle.support_native_out = True
