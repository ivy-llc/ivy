"""
MXNet device functions.

Collection of MXNet general functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
from typing import Union, Optional
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


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
    raise NotImplementedError("mxnet.to_device Not Implemented")


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
    raise NotImplementedError("mxnet.clear_cached_mem_on_dev Not Implemented")


def num_gpus() -> int:
    raise NotImplementedError("mxnet.num_gpus Not Implemented")


def gpu_is_available() -> bool:
    if mx.context.num_gpus() > 0:
        return True
    return False


def tpu_is_available() -> bool:
    raise NotImplementedError("mxnet.tpu_is_available Not Implemented")


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        raise NotImplementedError("mxnet.__init__ Not Implemented")

    def start(self):
        raise NotImplementedError("mxnet.start Not Implemented")

    def stop(self):
        raise NotImplementedError("mxnet.stop Not Implemented")

    def __enter__(self):
        raise NotImplementedError("mxnet.__enter__ Not Implemented")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("mxnet.__exit__ Not Implemented")
