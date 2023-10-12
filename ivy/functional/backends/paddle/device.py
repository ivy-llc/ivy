"""Collection of Paddle general functions, wrapped to fit Ivy syntax and
signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import (
    _shift_native_arrays_on_default_device,
    _as_ivy_dev_helper,
    _as_native_dev_helper,
    Profiler as BaseProfiler,
)
from paddle.device import core


# API #
# ----#


def dev(
    x: paddle.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, core.Place]:
    return x.place if as_native else ivy.as_ivy_dev(x.place)


def to_device(
    x: paddle.Tensor,
    device: core.Place,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    device = ivy.as_native_dev(device)
    if device.is_cpu_place():
        return x.cpu()
    elif device.is_gpu_place():
        return x.cuda(device.gpu_device_id())


def get_native_device_platform_and_id(device, /):
    if device.is_cpu_place():
        return "cpu", 0
    elif device.is_gpu_place():
        return "gpu", device.gpu_device_id()


def get_native_device(device_platform, device_id, /):
    native_dev = core.Place()
    if device_platform == "cpu":
        native_dev.set_place(paddle.device.core.CPUPlace())
    else:
        native_dev.set_place(paddle.device.core.CUDAPlace(device_id))

    return native_dev


def as_ivy_dev(device, /):
    return _as_ivy_dev_helper(device)


def as_native_dev(device, /):
    return _as_native_dev_helper(device)


def clear_mem_on_dev(device: core.Place, /):
    device = ivy.as_native_dev(device)
    if device.is_gpu_place():
        paddle.device.cuda.empty_cache()


def clear_cached_mem_on_dev(device: str, /):
    device = ivy.as_native_dev(device)
    if device.is_gpu_place():
        paddle.device.cuda.empty_cache()


def num_gpus() -> int:
    return paddle.device.cuda.device_count()


def gpu_is_available() -> bool:
    return bool(paddle.device.cuda.device_count())


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    return False


def handle_soft_device_variable(*args, fn, **kwargs):
    args, kwargs, device_shifting_dev = _shift_native_arrays_on_default_device(
        *args, **kwargs
    )
    # since there is no context manager for device in Paddle,
    # we need to manually set the device
    # then set it back to prev default device after the function call
    prev_def_dev = paddle.get_device()
    paddle.device.set_device(ivy.as_ivy_dev(device_shifting_dev))
    ret = fn(*args, **kwargs)
    paddle.device.set_device(ivy.as_ivy_dev(prev_def_dev))
    return ret


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        # ToDO: add proper Paddle profiler
        super(Profiler, self).__init__(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        time_taken = time.perf_counter() - self._start_time
        with open(os.path.join(self._save_dir, "profile.log"), "w+") as f:
            f.write(f"took {time_taken} seconds to complete")

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
