# global
from typing import Optional, Union, Sequence
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)


# dirichlet
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def dirichlet(
    alpha: Union[torch.tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    size = size if size is not None else len(alpha)
    if seed is not None:
        torch.manual_seed(seed)
    return torch.tensor(
        torch.distributions.dirichlet.Dirichlet(alpha).rsample(sample_shape=size),
        dtype=dtype,
    )


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, backend_version)
def beta(
    alpha: Union[float, torch.Tensor],
    beta: Union[float, torch.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[torch.dtype, ivy.Dtype]] = None,
    device: torch.device = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    if seed is not None:
        torch.manual_seed(seed)
    ret = torch.distributions.beta.Beta(alpha, beta).sample(shape)
    if device is not None:
        return ret.to(device)
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, backend_version)
def gamma(
    alpha: Union[float, torch.Tensor],
    beta: Union[float, torch.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[torch.dtype, ivy.Dtype]] = None,
    device: torch.device = None,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    if seed is not None:
        torch.manual_seed(seed)
    ret = torch.distributions.gamma.Gamma(alpha, beta).sample(shape)
    if device is not None:
        return ret.to(device)
    return ret


def _poisson_with_neg_lam(lam, fill_value, device, dtype):
    if torch.any(lam < 0):
        pos_lam = torch.where(lam < 0, 0, lam)
        ret = torch.poisson(pos_lam).type(dtype).to(device)
        ret = torch.where(lam < 0, fill_value, ret)
    else:
        ret = torch.poisson(lam).type(dtype).to(device)
    return ret


def poisson(
    lam: Union[float, torch.Tensor],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[torch.Tensor] = None,
):
    lam = torch.tensor(lam, device=device, dtype=torch.float32)
    if seed:
        torch.manual_seed(seed)
    if shape is None:
        return _poisson_with_neg_lam(lam, fill_value, device, dtype)
    shape = torch.tensor(shape, device=device, dtype=torch.int32)
    list_shape = shape.tolist()
    _check_shapes_broadcastable(lam.shape, list_shape)
    lam = torch.broadcast_to(lam, list_shape)
    return _poisson_with_neg_lam(lam, fill_value, device, dtype)


def bernoulli(
    probs: Union[float, torch.Tensor],
    *,
    logits: Union[float, torch.Tensor] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if seed:
        torch.manual_seed(seed)
    if logits is not None:
        if not _check_shapes_broadcastable(shape, logits.shape):
            shape = logits.shape
    elif probs is not None:
        if not _check_shapes_broadcastable(shape, probs.shape):
            shape = probs.shape
    return (
        torch.distributions.bernoulli.Bernoulli(probs=probs, logits=logits)
        .sample(shape)
        .to(device, dtype)
    )
