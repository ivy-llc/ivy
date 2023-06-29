"""Collection of Jax device functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import jax
from typing import Union, Optional
import jaxlib.xla_extension
import logging

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.device import Profiler as BaseProfiler


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
        logging.warn(
            "Cannot get device from DynamicJaxprTracer, returning empty string."
        )
        if as_native:  # fallback
            return jax.devices()[0]
        # TODO: Find the code where JaxprTracer device is required
        # and get a cleaner workaround for this issue
        return ""  # change might break something
    try:
        dv = _to_array(x).device()
    except Exception:
        dv = jax.devices()[0]
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_device(
    x: JaxArray,
    device: Union[jaxlib.xla_extension.Device, ivy.Device],
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[JaxArray] = None,
):
    # TODO: implement stream and out
    return _to_device(x, device=device)


# this is a non-wrapped function used to place JAX arrays on respective devices,
# since if we use to_device, it will return ivy.array which is not desirable
def _to_device(x, device=None):
    if device is not None:
        device = as_native_dev(device)
        cur_dev = dev(x, as_native=True)
        if cur_dev != device:
            # Uses async dispatch, data commited only when required
            x = jax.device_put(x, device)
    return x


def as_ivy_dev(device, /):
    # doesn't check if device is a valid device
    if isinstance(device, str):
        return ivy.Device(device)
    if is_native_dev(device):  # no duck typing
        p, dev_id = (device.platform, device.id)
        if p == "cpu" and dev_id == 0:
            return ivy.Device(p)
        return ivy.Device(p + ":" + str(dev_id))
    else:
        raise ivy.utils.exceptions.IvyException(
            f"Cannot convert {device} to an ivy device. Expected a "
            f"jaxlib.xla_extension.Device or str, got {type(device)}"
        )


def as_native_dev(device, /):
    # checks if device is a valid device internally
    if is_native_dev(device):
        return device
    if isinstance(device, str):
        dev_split = ivy.Device(device).split(":")
        device = dev_split[0]
        if len(dev_split) > 1:
            idx = int(dev_split[1])
        else:
            idx = 0
        existing_devices = jax.devices(device)
        return jax.devices(device)[min(idx, len(existing_devices) - 1)]
    else:
        raise ivy.utils.exceptions.IvyError(
            f"Cannot convert {device} to a JaX device. Expected a "
            f"jaxlib.xla_extension.Device or ivy.Device, got {type(device)}"
        )


def is_native_dev(device, /):
    return isinstance(device, jaxlib.xla_extension.Device)


def clear_cached_mem_on_dev(device: str, /):
    # Refer: https://github.com/google/jax/issues/1222 [updated 2023-04-29]
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
        return len(jax.devices("gpu"))
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
