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
    return as_ivy_dev(dv)


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


def is_valid_device(device_platform, device_id, /):
    return device_platform in ["gpu", "tpu"] and device_id in range(
        0, jax.device_count(device_platform)
    )


def as_ivy_dev(device, /):
    """
    Convert a JAX device to an ivy Device.

    Parameters
    ----------
    device
        The JAX device to convert.
        If the device is a string, it must be in the format
        'platform:id'.

    If the device is a JAX device, the platform and id are extracted.
    If the device is None, the default device is returned.
    returns
    -------
    ret
        The converted ivy Device.
    Examples
    --------
    >>> ivy.as_ivy_dev(jax.devices()[0])
    Device(cpu)
    >>> ivy.as_ivy_dev('gpu:0')
    Device(gpu:0)
    >>> ivy.as_ivy_dev(None)
    Device(cpu)
    >>> ivy.as_ivy_dev(jax.devices()[1])
    Device(cpu:1)
    >>> ivy.as_ivy_dev('gpu:1')
    Device(gpu:1)
    >>> ivy.as_ivy_dev('tpu:0')
    Device(tpu:0)
    >>> ivy.as_ivy_dev('tpu:1')
    Device(tpu:1)
    >>> ivy.as_ivy_dev('tpu:2')
    Device(tpu:2)
    >>> ivy.as_ivy_dev('tpu:3')
    """
    if device is None:
        return ivy.Device("cpu")
    if isinstance(device, str):
        device = ivy.Device(device)
        p, dev_id = (device[0:3], int(device[4:]))
    else:
        p, dev_id = (device.platform, device.id)
    if p == "cpu":
        return ivy.Device(p)
    if is_valid_device(p, dev_id):
        return ivy.Device(p + ":" + str(dev_id))
    else:
        return ivy.Device(p + ":" + str(0))


def as_native_dev(device, /):
    """
    Convert an ivy Device to a JAX device.

    Parameters
    ----------
    device
        The ivy Device to convert.
    If the device is a string, it must be in the format
    'platform:id'.
    If the device is a Ivy device, it must be in the format
        'platform:id'.
    If the device is None, the default device is returned.
    returns
    -------
    ret
        The converted JAX device.
    Examples
    --------
    >>> ivy.as_native_dev(ivy.Device('cpu'))
    CpuDevice(id=0)
    """
    if isinstance(device, jaxlib.xla_extension.Device) and is_valid_device(
        device.platform, device.id
    ):
        return device
    dev_split = ivy.Device(device).split(":")
    device = dev_split[0]
    if len(dev_split) > 1:
        idx = int(dev_split[1])
    else:
        idx = 0

    if device == "cpu":
        return jax.devices(device)[idx]

    elif is_valid_device(device, idx):
        return jax.devices(device)[idx]
    else:
        return jax.devices(device)[0]


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
