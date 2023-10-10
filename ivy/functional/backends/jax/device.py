"""Collection of Jax device functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import jax
from typing import Union, Optional
import jaxlib.xla_extension

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.device import (
    _shift_native_arrays_on_default_device,
    _get_device_platform_and_id,
    Profiler as BaseProfiler,
)


# Helpers #
# --------#


def _to_array(x):
    if isinstance(x, jax.interpreters.ad.JVPTracer):
        return _to_array(x.primal)
    elif isinstance(x, jax.interpreters.partial_eval.DynamicJaxprTracer):
        return _to_array(x.aval)
    elif isinstance(x, jax.interpreters.batching.BatchTracer):
        return _to_array(x.val)
    return x


# API #
# ----#


def dev(
    x: JaxArray,
    /,
    *,
    as_native: bool = False,
) -> Union[ivy.Device, jaxlib.xla_extension.Device]:
    if isinstance(x, jax.interpreters.partial_eval.DynamicJaxprTracer):
        return ""
    try:
        dv = _to_array(x).device_buffer.device
        dv = dv()
    except Exception:
        dv = jax.devices()[0]
    if as_native:
        return dv
    return ivy.as_ivy_dev(dv)


def to_device(
    x: JaxArray,
    device: jaxlib.xla_extension.Device,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[JaxArray] = None,
):
    if device is not None:
        cur_dev = as_native_dev(dev(x))
        if cur_dev != device:
            x = jax.device_put(x, as_native_dev(device))
    return x


# this is a non-wrapped function used to place JAX arrays on respective devices,
# since if we use to_device, it will return ivy.array which is not desirable
def _to_device(x, device=None):
    if device is not None:
        cur_dev = as_native_dev(dev(x))
        if cur_dev != device:
            x = jax.device_put(x, as_native_dev(device))
    return x


def _is_valid_device(device_platform, device_id, /):
    return device_platform in ["gpu"] and device_id in range(
        0,
    )


# def as_ivy_dev(device, /):
#     if isinstance(device, str):
#         device_platform, device_id = _get_device_platform_and_id(device)
#     elif isinstance(device, ivy.NativeDevice):
#         device_platform, device_id = (device.platform, device.id)
#     else:
#         raise ivy.exceptions.IvyDeviceError(
#             "Device is not supported or the format is wrong!"
#         )
#     if device_platform in [None, "cpu"]:
#         return ivy.Device("cpu")
#     if _is_valid_device(device_platform, device_id):
#         return ivy.Device(f"{device_platform}:{device_id}")
#     else:
#         return ivy.Device(f"{device_platform}:{0}")


def as_native_dev(device, /):
    if isinstance(device, jaxlib.xla_extension.Device):
        return device
    elif isinstance(device, str):
        device_platform, device_id = _get_device_platform_and_id(device)
    else:
        raise ivy.exceptions.IvyDeviceError(
            "Device is not supported or the format is wrong!"
        )
    if device_platform in [None, "cpu"]:
        return jax.devices("cpu")[0]
    if _is_valid_device(device_platform, device_id):
        return jax.devices(device_platform)[device_id]
    else:
        return jax.devices(device_platform)[0]


def handle_soft_device_variable(*args, fn, **kwargs):
    args, kwargs, device_shifting_dev = _shift_native_arrays_on_default_device(
        *args, **kwargs
    )
    with jax.default_device(device_shifting_dev):
        return fn(*args, **kwargs)


def clear_cached_mem_on_dev(device: str, /):
    return None


def _dev_is_available(base_dev):
    try:
        jax.devices(base_dev)
        return True
    except RuntimeError:
        return False


def gpu_is_available() -> bool:
    return _dev_is_available("gpu")


def num_gpus() -> int:
    try:
        return jax.device_count("gpu")
    except RuntimeError:
        return 0


def tpu_is_available() -> bool:
    return _dev_is_available("tpu")


# noinspection PyMethodMayBeStatic
class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        super(Profiler, self).__init__(save_dir)
        self._save_dir = os.path.join(self._save_dir, "profile")

    def start(self):
        jax.profiler.start_trace(self._save_dir)

    def stop(self):
        jax.profiler.stop_trace()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
