"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import Optional, List, Union, Tuple, Sequence

# local
from ivy.functional.ivy.device import default_device


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype=None,
    *,
    device: torch.device
) -> torch.Tensor:
    rand_range = high - low
    if shape is None:
        shape = []
    return (
        torch.rand(shape, device=default_device(device), dtype=dtype) * rand_range + low
    )


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[List[int]] = None,
    *,
    device: torch.device
) -> torch.Tensor:
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean
    std = std.item() if isinstance(std, torch.Tensor) else std
    return torch.normal(mean, std, true_shape, device=default_device(device))


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[torch.Tensor] = None,
    replace: bool = True,
    *,
    device: torch.device
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
    return torch.multinomial(probs.float(), num_samples, replace).to(
        default_device(device)
    )


def randint(
    low: int,
    high: int,
    shape: Union[int, Sequence[int]],
    *,
    device: torch.device,
    out: torch.Tensor,
) -> torch.Tensor:
    return torch.randint(low, high, shape, device=default_device(device))


def seed(seed_value: int = 0) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size))
