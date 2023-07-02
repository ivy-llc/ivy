"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.fluid.libpaddle import Place
from paddle.fluid import core
from paddle.framework import _get_paddle_place

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
) -> Union[ivy.Device, _paddle_dev_types]:
    dv = x.place
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_device(
    x: paddle.Tensor,
    device: Union[ivy.Device, _paddle_dev_types],
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if device is not None:
        native_dev = as_native_dev(device)
        if not dev(x, as_native=True)._equals(native_dev):  # type: ignore
            # TODO: check memory leak because ret is copying the original tensor
            return paddle.to_tensor(x, place=native_dev, stop_gradient=x.stop_gradient)
    return x


def as_ivy_dev(
    device: Union[str, _paddle_dev_types],
    /,
) -> ivy.Device:
    if isinstance(device, str):
        return ivy.Device(device)

    if is_native_dev(device):
        if not isinstance(device, Place):
            native_dev = Place()
            native_dev.set_place(device)
        else:
            native_dev = device
        if native_dev.is_cpu_place():
            return ivy.Device("cpu")
        elif native_dev.is_gpu_place():
            dev_type = "gpu"
            dev_idx = device.gpu_device_id()
            return ivy.Device(dev_type + ":" + str(dev_idx))
        elif native_dev.is_gpu_pinned_place():
            return ivy.Device("gpu:0")  # simiplification
        else:
            raise ivy.utils.exceptions.IvyError(
                f"Cannot convert {device} to an ivyDevice. ",
                "Currently only cpu, tpu and gpu are supported.",
            )

    raise ivy.utils.exceptions.IvyError(
        f"Cannot convert {device} to an ivy device. Expected a "
        f"paddle.fluid.libpaddle.Place or str, got {type(device)}"
    )


def as_native_dev(
    device: Union[ivy.Device, str, _paddle_dev_types],
    /,
) -> _paddle_dev_types:
    if is_native_dev(device):
        return device
    if isinstance(device, str):
        if "tpu" in device:
            raise ivy.utils.exceptions.IvyBackendException(
                "TPU not supported in paddle backend. Consider using ",
                "TensorFlow or JaX.",
            )
        else:
            # Internal function handles more edge cases and devices
            return _get_paddle_place(device)
    else:
        raise ivy.utils.exceptions.IvyError(
            f"{device} cannot be converted to paddle native device."
        )


def is_native_dev(device, /):
    return isinstance(device, _paddle_dev_types.__args__)


def clear_mem_on_dev(device: Place, /):
    try:
        # throws and error if not compiled with cuda
        device = as_native_dev(device)
        if (
            isinstance(device, Place)
            and device.is_cuda_place()
            or isinstance(device, (core.CUDAPlace, core.CUDAPinnedPlace))
        ):
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


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
