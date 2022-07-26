"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import Optional, Union, Sequence
import numbers

# local
import ivy
from ivy.functional.ivy.random import _check_bounds_and_get_shape

# Extra #
# ------#


def random_uniform(
    low: Union[float, torch.Tensor] = 0.0,
    high: Union[float, torch.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shape = _check_bounds_and_get_shape(low, high, shape)
    rand_range = high - low
    return torch.rand(shape, device=device, dtype=dtype, out=out) * rand_range + low


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
    return torch.multinomial(probs.float(), num_samples, replace, out=out).to(device)


multinomial.support_native_out = True


def randint(
    low: Union[int, torch.Tensor],
    high: Union[int, torch.Tensor],
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    zero_dim = len(shape) == 0
    if zero_dim:
        shape = [1]
    ret = torch.rand(*shape, out=out, dtype=torch.float64, device=device)
    ret = torch.mul(ret, high - low, out=out)
    ret = torch.add(ret, low, out=out)
    ret = ret.to(ivy.default_int_dtype(as_native=True))
    ret = torch.clamp(ret, low, high - 1)
    if zero_dim:
        return ret.reshape(())
    return ret


randint.support_native_out = True


def seed(seed_value: int = 0) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size, out=out), out=out)


shuffle.support_native_out = True
