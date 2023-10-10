"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import (
    _shift_native_arrays_on_default_device,
    _get_device_platform_and_id,
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
    device = as_native_dev(device)
    if device.is_cpu_place():
        return x.cpu()
    elif device.is_gpu_place():
        return x.cuda(device.gpu_device_id())


def _is_valid_device(device_platform, device_id, /):
    return device_platform in ["gpu"] and device_id in range(0, num_gpus())


# def as_ivy_dev(device: core.Place, /):
#     if isinstance(device, str):
#         device_platform, device_id = _get_device_platform_and_id(device)
#     elif device is None or device.is_cpu_place():
#         return ivy.Device("cpu")
#     elif isinstance(device, ivy.NativeDevice):
#         device_platform, device_id = ("gpu", device.gpu_device_id())
#     else:
#         raise ivy.exceptions.IvyDeviceError(
#             "Device is not supported or the format is wrong!"
#         )

#     if _is_valid_device(device_platform, device_id):
#         return ivy.Device(f"{device_platform}:{device_id}")
#     else:
#         return ivy.Device(f"{device_platform}:{0}")


def as_native_dev(
    device: Optional[Union[ivy.Device, core.Place]] = None,
    /,
) -> core.Place:
    if isinstance(device, core.Place):
        return device
    elif isinstance(device, str):
        device_platform, device_id = _get_device_platform_and_id(device)
    else:
        raise ivy.exceptions.IvyDeviceError(
            "Device is not supported or the format is wrong!"
        )
    native_dev = core.Place()
    if device_platform in [None, "cpu"]:
        native_dev.set_place(paddle.device.core.CPUPlace())

    if _is_valid_device(device_platform, device_id):
        native_dev.set_place(paddle.device.core.CUDAPlace(device_id))
        return native_dev
    else:
        native_dev.set_place(paddle.device.core.CUDAPlace(0))
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
