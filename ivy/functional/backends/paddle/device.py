"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.device import core


# API #
# ----#


def dev(
    x: paddle.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, core.Place]:
    return x.place if as_native else as_ivy_dev(x.place)


def to_device(
    x: paddle.Tensor,
    device: core.Place,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    device = as_native_dev(device)
    if device.is_cpu_place():
        return x.cpu()
    elif device.is_gpu_place():
        return x.cuda(device.gpu_device_id())


def as_ivy_dev(device: core.Place, /):
    # TODO: add handling to string inputs without indices for gpu
    if isinstance(device, str):
        return ivy.Device(device)

    # TODO: remove this once ivy.Device accepts native device inputs
    if device.is_cpu_place():
        return ivy.Device("cpu")
    elif device.is_gpu_place():
        dev_idx = device.gpu_device_id()
        return ivy.Device("gpu:" + str(dev_idx))


def as_native_dev(
    device: Optional[Union[ivy.Device, core.Place]] = None,
    /,
) -> core.Place:
    if isinstance(device, core.Place):
        return device
    native_dev = core.Place()
    if "cpu" in device:
        native_dev.set_place(paddle.device.core.CPUPlace())

    elif "gpu" in device:
        if ":" in device:
            gpu_idx = int(device.split(":")[-1])
            assert (
                gpu_idx < num_gpus()
            ), "The requested device is higher than the number of available devices"
        else:
            gpu_idx = 0
        native_dev.set_place(paddle.device.core.CUDAPlace(gpu_idx))
    return native_dev


def clear_mem_on_dev(device: core.Place, /):
    device = as_native_dev(device)
    if device.is_gpu_place():
        paddle.device.cuda.empty_cache()


def clear_cached_mem_on_dev(device: str, /):
    device = as_native_dev(device)
    if device.is_gpu_place():
        paddle.device.cuda.empty_cache()


def num_gpus() -> int:
    return paddle.device.cuda.device_count()


def gpu_is_available() -> bool:
    return bool(paddle.device.cuda.device_count())


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    return False


def handle_soft_device_variable(*args, fn, dst_dev=None, **kwargs):
    args, kwargs, dst_dev = ivy.shift_native_arrays_on_def_dev(
        *args, dst_dev=dst_dev, **kwargs
    )
    # since there is no context manager for device in Paddle, we need to manually set the device
    # then set it back to prev default device after the function call
    prev_def_dev = paddle.get_device()
    paddle.device.set_device(ivy.as_ivy_dev(dst_dev))
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
            f.write("took {} seconds to complete".format(time_taken))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
