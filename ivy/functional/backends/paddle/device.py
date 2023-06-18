"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.fluid.libpaddle import Place


# API #
# ----#


def dev(x: paddle.Tensor, /, *, as_native: bool = False) -> Union[ivy.Device, Place]:
    dv = x.place
    if as_native:
        if isinstance(dv, Place):
            dv = "gpu" if dv.is_gpu_place() else "cpu"
        return x.place
    return as_ivy_dev(dv)


def to_device(
    x: paddle.Tensor,
    device: Place,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if device is None:
        return x
    # TODO: check memory leak because ret is copying the original tensor
    ret = paddle.to_tensor(
        x, place=as_native_dev(device), stop_gradient=x.stop_gradient
    )
    return ret


def as_ivy_dev(device: Place, /):
    if isinstance(device, str):
        return ivy.Device(device)

    if device.is_cpu_place():
        dev_type = "cpu"
        dev_idx = 0
    elif device.is_gpu_place():
        dev_type = "gpu"
        dev_idx = device.gpu_device_id()

    if dev_type == "cpu":
        return ivy.Device(dev_type)

    return ivy.Device(dev_type + (":" + (str(dev_idx) if dev_idx is not None else "0")))


def as_native_dev(
    device: Optional[Union[ivy.Device, Place]] = None,
    /,
) -> Optional[Place]:
    if not isinstance(device, str):
        return device
    elif "cpu" in device:
        return paddle.fluid.libpaddle.CPUPlace()
    elif "gpu" in device:
        return paddle.fluid.libpaddle.CUDAPlace()


def clear_mem_on_dev(device: Place, /):
    device = as_native_dev(device)
    if isinstance(device, paddle.fluid.libpaddle.CUDAPlace):
        paddle.device.cuda.empty_cache()


def clear_cached_mem_on_dev(device: str, /):
    if "gpu" in device:
        paddle.device.cuda.empty_cache()
    return None


def num_gpus() -> int:
    return paddle.device.cuda.device_count()


def gpu_is_available() -> bool:
    return bool(paddle.device.cuda.device_count())


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    return False


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
