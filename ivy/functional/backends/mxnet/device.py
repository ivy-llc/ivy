"""Collection of MXNet general functions, wrapped to fit Ivy syntax and signature."""

# global
import os

_round = round
import mxnet as mx
from typing import Union
from mxnet import profiler as _profiler

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(
    x: mx.nd.NDArray, as_native: bool = False
) -> Union[ivy.Device, mx.context.Context]:
    dv = x.context
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_dev(x, device=None, out=None):
    if device is not None:
        ret = x.as_in_context(as_native_dev(device))
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x


def as_ivy_dev(device):
    if isinstance(device, str):
        return ivy.Device(device)
    device_type = device.device_type
    if device_type == "cpu":
        return ivy.Device(device_type)
    return ivy.Device(
        device_type
        + (":" + (str(device.device_id) if device.device_id is not None else "0"))
    )


def as_native_dev(device):
    if not isinstance(device, str):
        return device
    dev_split = ivy.Device(device).split(":")
    device = dev_split[0]
    if len(dev_split) > 1:
        idx = int(dev_split[1])
    else:
        idx = 0
    return mx.context.Context(device, idx)


def gpu_is_available() -> bool:
    return mx.context.num_gpus() > 0


clear_mem_on_dev = lambda device: None
_callable_dev = dev


def tpu_is_available() -> bool:
    return False


def num_gpus() -> int:
    return mx.context.num_gpus()


class Profiler(BaseProfiler):
    def __init__(self, save_dir):
        super(Profiler, self).__init__(save_dir)
        self._prof = _profiler
        self._prof.set_config(
            profile_all=True,
            aggregate_stats=True,
            continuous_dump=True,
            filename=os.path.join(save_dir, "trace.json"),
        )

    def start(self):
        self._prof.set_state("run")

    def stop(self):
        self._prof.set_state("stop")
        self._prof.dump()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
