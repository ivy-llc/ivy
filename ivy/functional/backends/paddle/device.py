"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.fluid.libpaddle import Place, CPUPlace, CUDAPinnedPlace, CUDAPlace


# API #
# ----#


def dev(x: paddle.Tensor, /, *, as_native: bool = False) -> Union[ivy.Device, Place]:
    dv = x.place
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_device(
    x: paddle.Tensor,
    device: Union[ivy.Device, Place],
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if device is not None:
        # the string should be able to be converted to paddle.fluid.libpaddle.Place
        ivy_dev = as_ivy_dev(device)
        if dev(x, as_native=False) != ivy_dev:
            # TODO: check memory leak because ret is copying the original tensor
            return paddle.to_tensor(x, place=ivy_dev, stop_gradient=x.stop_gradient)
    return x


def as_ivy_dev(device, /):
    if isinstance(device, str):
        return ivy.Device(device)

    if is_native_dev(device):
        if device.is_cpu_place():
            return ivy.Device("cpu")
        elif device.is_gpu_place():
            dev_type = "gpu"
            dev_idx = device.gpu_device_id()
            return ivy.Device(dev_type + ":" + str(dev_idx))

    raise ivy.utils.exceptions.IvyBackendException(
        f"Cannot convert {device} to an ivy device. Expected a "
        f"paddel.fluid.libpaddle.Place or str, got {type(device)}"
    )


def as_native_dev(
    device: Optional[Union[ivy.Device, Place]] = None,
    /,
) -> Optional[Place]:
    if is_native_dev(device):
        return device
    if isinstance(device, str):
        # if device[:3] in ["cpu", "gpu"]:
        if "cpu" in device:
            return paddle.fluid.libpaddle.CPUPlace()
        elif "gpu" in device:
            return paddle.fluid.libpaddle.CUDAPlace()
        elif "tpu" in device:
            raise ivy.utils.exceptions.IvyBackendException(
                "TPU not supported in Paddle backend."
            )
        else:
            raise ivy.utils.exceptions.IvyException(
                f"{device} cannot be converted to paddle native device."
            )


def is_native_dev(device, /):
    # TODO: check if only paddle.fluid.libpaddle.Place is needed
    return isinstance(device, (Place, CPUPlace, CUDAPlace, CUDAPinnedPlace))


def clear_mem_on_dev(device: Place, /):
    device = as_native_dev(device)
    if isinstance(device, paddle.fluid.libpaddle.CUDAPlace):
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
