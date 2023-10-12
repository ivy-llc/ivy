"""Collection of Jax device functions, wrapped to fit Ivy syntax and
signature."""

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
    _as_ivy_dev_helper,
    _as_native_dev_helper,
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
        cur_dev = ivy.as_native_dev(dev(x))
        if cur_dev != device:
            x = jax.device_put(x, ivy.as_native_dev(device))
    return x


# this is a non-wrapped function used to place JAX arrays on respective devices,
# since if we use to_device, it will return ivy.array which is not desirable
def _to_device(x, device=None):
    if device is not None:
        cur_dev = ivy.as_native_dev(dev(x))
        if cur_dev != device:
            x = jax.device_put(x, ivy.as_native_dev(device))
    return x


def get_native_device_platform_and_id(device, /):
    return (device.platform, device.id)


def get_native_device(device_platform, device_id, /):
    return jax.devices(device_platform)[device_id]


def as_ivy_dev(device, /):
    return _as_ivy_dev_helper(device)


def as_native_dev(device, /):
    return _as_native_dev_helper(device)


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
