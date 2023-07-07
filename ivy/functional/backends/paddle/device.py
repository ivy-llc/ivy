"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.device import core

_paddle_dev_types = Union[
    core.Place,
    core.XPUPlace,
    core.CPUPlace,
    core.CUDAPinnedPlace,
    core.CUDAPlace,
    core.IPUPlace,
    core.CustomPlace,
]


# API #
# ----#


def dev(
    x: paddle.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, core.Place]:
    return x.place if as_native else as_ivy_dev(x.place)


def to_device(
    x: paddle.Tensor,
    device: _paddle_dev_types,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    device = as_native_dev(device)
    if device.is_cpu_place():
        ret = x.cpu()
    elif device.is_gpu_place():
        ret = x.cuda(device.gpu_device_id())
    else:
        ret = x
    if out:
        return paddle.assign(ret, output=out)
    return ret


def as_ivy_dev(device: _paddle_dev_types, /):
    if isinstance(device, str):
        return ivy.Device(device)

    if is_native_dev(device):
        if not isinstance(device, core.Place):
            native_dev = core.Place()
            native_dev.set_place(device)
        else:
            native_dev = device
        if native_dev.is_cpu_place():
            return ivy.Device("cpu")
        elif native_dev.is_gpu_place():
            dev_idx = device.gpu_device_id()
            return ivy.Device("gpu:" + str(dev_idx))
        elif native_dev.is_gpu_pinned_place():
            return ivy.Device("gpu:0")  # simiplification
        else:
            raise NotImplementedError(f"Device type {device} not supported in ivy.")

    raise ivy.utils.exceptions.IvyException(
        f"Cannot convert {device} to an ivy device. Expected a "
        f"paddle Device or str, got {type(device)}"
    )


def as_native_dev(
    device: Optional[Union[ivy.Device, _paddle_dev_types]] = None,
    /,
) -> core.Place:
    native_dev = core.Place()
    if is_native_dev(device):
        if isinstance(device, core.Place):
            return device
        native_dev = core.Place()
        native_dev.set_place(device)
        return native_dev

    if isinstance(device, str):
        device = device.lower()
        if "cpu" in device:
            native_dev.set_place(paddle.device.core.CPUPlace())

        elif "gpu" in device:
            if not core.is_compiled_with_cuda():
                raise ValueError(
                    "The device should not be gpu, since PaddlePaddle is "
                    "not compiled with CUDA"
                )
            if ":" in device:
                gpu_idx = device.split(":")[-1]
                ivy.utils.assertions.check_true(
                    gpu_idx.isnumeric(),
                    message=f"{gpu_idx} must be numeric",
                )
                gpu_idx = min(int(gpu_idx), num_gpus() - 1)
            else:
                gpu_idx = 0
            native_dev.set_place(paddle.device.core.CUDAPlace(gpu_idx))
        return native_dev

    raise ivy.utils.exceptions.IvyException(
        f"Cannot convert {device} to an ivy device. Expected a "
        f"paddle Device or str, got {type(device)}"
    )


def is_native_dev(device, /):
    return isinstance(device, _paddle_dev_types.__args__)


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
