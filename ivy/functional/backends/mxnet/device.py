"""
MXNet device functions.

Collection of MXNet general functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
from typing import Union, Optional
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from ivy.utils.exceptions import IvyNotImplementedException


def dev(
    x: Union[(None, mx.ndarray.NDArray)], /, *, as_native: bool = False
) -> Union[(ivy.Device, str)]:
    if as_native:
        return x.context
    return as_ivy_dev(x.context)


def to_device(
    x: Union[(None, mx.ndarray.NDArray)],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return x.as_in_context(as_native_dev(device))


def as_ivy_dev(device):
    if isinstance(device, str):
        return ivy.Device(device)
    if device is None:
        return None
    # if mx device context is passed
    p, dev_id = (device.device_type, device.device_id)
    if p == "cpu":
        return ivy.Device(p)
    return ivy.Device(p + ":" + str(dev_id))


def as_native_dev(device: str, /):
    if isinstance(device, mx.Context):
        return device
    if device is None or device.find("cpu") != -1:
        mx_dev = "cpu"
    elif device.find("gpu") != -1:
        mx_dev = "gpu"
    else:
        raise Exception("dev input {} not supported.".format(device))
    if device.find(":") != -1:
        mx_dev_id = int(device[device.find(":") + 1 :])
    else:
        mx_dev_id = 0
    return mx.Context(mx_dev, mx_dev_id)


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
