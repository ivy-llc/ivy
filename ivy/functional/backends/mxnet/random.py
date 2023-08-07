"""
MXNet random functions.

Collection of MXNet random functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
from typing import Optional, Union, Sequence
import ivy

from ivy.utils.exceptions import IvyNotImplementedException


def random_uniform(
    *,
    low: Union[(float, None, mx.ndarray.NDArray)] = 0.0,
    high: Union[(float, None, mx.ndarray.NDArray)] = 1.0,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int], None)]] = None,
    dtype: None,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def random_normal(
    *,
    mean: Union[(float, None, mx.ndarray.NDArray)] = 0.0,
    std: Union[(float, None, mx.ndarray.NDArray)] = 1.0,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    dtype: None,
    seed: Optional[int] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    replace: bool = True,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def randint(
    low: Union[(float, None, mx.ndarray.NDArray)],
    high: Union[(float, None, mx.ndarray.NDArray)],
    /,
    *,
    shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: Optional[Union[(None, ivy.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def seed(*, seed_value: int = 0) -> None:
    raise IvyNotImplementedException()


def shuffle(
    x: Union[(None, mx.ndarray.NDArray)],
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
