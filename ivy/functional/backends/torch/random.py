"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import Optional, Union, Sequence
import numbers

# local
import ivy
from ivy.functional.ivy.device import default_device


# Extra #
# ------#


def random_uniform(
    low: Union[float, torch.Tensor] = 0.0,
    high: Union[float, torch.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype=None,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rand_range = high - low
    if shape is None:
        shape = []
    return (
        torch.rand(shape, device=default_device(device), dtype=dtype, out=out)
        * rand_range
        + low
    )


random_uniform.support_native_out = True


def random_normal(
    mean: Union[float, torch.Tensor] = 0.0,
    std: Union[float, torch.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(mean, numbers.Number) and isinstance(std, numbers.Number):
        ret = torch.normal(mean, std, ivy.default(shape, ()), out=out)
    else:
        assert shape is None, (
            "can only provide explicit shape if mean and std are " "both scalar values"
        )
        ret = torch.normal(mean, std, out=out)
    if ret.device == device:
        return ret
    return ret.to(device)


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


def seed(seed_value: int = 0) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size, out=out), out=out)


shuffle.support_native_out = True
