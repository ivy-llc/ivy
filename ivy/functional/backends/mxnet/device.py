"""
MXNet device functions.

Collection of MXNet general functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
from typing import Union, Optional
import ivy
from ivy.functional.ivy.device import (
    _get_device_platform_and_id,
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
    return x.as_in_context(as_native_dev(device))


def _is_valid_device(device_plateform, device_id, /):
    return device_plateform in ["gpu"] and device_id in range(0, mx.context.num_gpus())


# def as_ivy_dev(device):
#     if isinstance(device, str):
#         device_platform, device_id = _get_device_platform_and_id(device)
#     elif isinstance(device, ivy.NativeDevice):
#         device_platform, device_id = (device.device_type, device.device_id)
#     else:
#         raise ivy.exceptions.IvyDeviceError(
#             "Device is not supported or the format is wrong!"
#         )
#     if device_platform in [None, "cpu"]:
#         return ivy.Device("cpu")
#     if _is_valid_device(device_platform, device_id):
#         return ivy.Device(f"{device_platform}:{device_id}")
#     else:
#         return ivy.Device(f"{device_platform}:{0}")


def as_native_dev(device: str, /):
    if isinstance(device, mx.Context):
        return device
    elif isinstance(device, str):
        device_platform, device_id = _get_device_platform_and_id(device)
    else:
        raise ivy.exceptions.IvyDeviceError(
            "Device is not supported or the format is wrong!"
        )
    if device_platform in [None, "cpu"]:
        return mx.Context("cpu", device_id)
    if _is_valid_device(device_platform, device_id):
        return mx.Context(device_platform, device_id)
    else:
        return mx.Context(device_platform, 0)


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
