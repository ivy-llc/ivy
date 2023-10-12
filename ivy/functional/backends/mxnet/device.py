"""MXNet device functions.

Collection of MXNet general functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
from typing import Union, Optional
import ivy
from ivy.functional.ivy.device import (
    _as_ivy_dev_helper,
    _as_native_dev_helper,
    Profiler as BaseProfiler,
)
from ivy.utils.exceptions import IvyNotImplementedException


def dev(
    x: Union[(None, mx.ndarray.NDArray)], /, *, as_native: bool = False
) -> Union[(ivy.Device, str)]:
    if as_native:
        return x.context
    return ivy.as_ivy_dev(x.context)


def to_device(
    x: Union[(None, mx.ndarray.NDArray)],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return x.as_in_context(ivy.as_native_dev(device))


def get_native_device_platform_and_id(device, /):
    return (device.device_type, device.device_id)


def get_native_device(device_platform, device_id, /):
    return mx.Context(device_platform, device_id)


def as_ivy_dev(device, /):
    return _as_ivy_dev_helper(device)


def as_native_dev(device, /):
    return _as_native_dev_helper(device)


def clear_cached_mem_on_dev(device: str, /):
    raise IvyNotImplementedException()


def num_gpus() -> int:
    return mx.context.num_gpus()


def gpu_is_available() -> bool:
    if mx.context.num_gpus() > 0:
        return True
    return False


def tpu_is_available() -> bool:
    return False


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        raise IvyNotImplementedException()

    def start(self):
        raise IvyNotImplementedException()

    def stop(self):
        raise IvyNotImplementedException()

    def __enter__(self):
        raise IvyNotImplementedException()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise IvyNotImplementedException()
