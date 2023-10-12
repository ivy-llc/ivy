"""Tensorflow device functions.

Collection of TensorFlow general functions, wrapped to fit Ivy syntax
and signature.
"""

# global
_round = round
import tensorflow as tf
from typing import Union, Optional

# local
import ivy
from ivy.functional.ivy.device import (
    _as_ivy_dev_helper,
    _as_native_dev_helper,
    Profiler as BaseProfiler,
)


def _same_device(dev_a, dev_b):
    if dev_a is None or dev_b is None:
        return False
    return "/" + ":".join(dev_a[1:].split(":")[-2:]) == "/" + ":".join(
        dev_b[1:].split(":")[-2:]
    )


def dev(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    as_native: bool = False,
) -> Union[ivy.Device, str]:
    dv = x.device
    if as_native:
        return dv
    return ivy.as_ivy_dev(dv)


def to_device(
    x: Union[tf.Tensor, tf.Variable],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if device is None:
        return x
    device = ivy.as_native_dev(device)
    current_dev = dev(x)
    if not _same_device(current_dev, device):
        with tf.device("/" + device.upper()):
            return tf.identity(x)
    return x


def get_native_device_platform_and_id(device, /):
    dev_in_split = device[1:].split(":")[-2:]
    device_platform, device_id = dev_in_split[0].lower(), int(dev_in_split[1])
    return (device_platform, device_id)


def get_native_device(device_platform, device_id, /):
    return f"/{device_platform.upper()}:{device_id}"


def as_ivy_dev(device, /):
    return _as_ivy_dev_helper(device)


def as_native_dev(device, /):
    return _as_native_dev_helper(device)


def clear_cached_mem_on_dev(device: str, /):
    return None


def num_gpus() -> int:
    return len(tf.config.list_physical_devices("GPU"))


def gpu_is_available() -> bool:
    return len(tf.config.list_physical_devices("GPU")) > 0


def tpu_is_available() -> bool:
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tf.config.list_logical_devices("TPU")
        tf.distribute.experimental.TPUStrategy(resolver)
        return True
    except ValueError:
        return False


def handle_soft_device_variable(*args, fn, **kwargs):
    with tf.device(ivy.default_device(as_native=True)):
        return fn(*args, **kwargs)


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        super(Profiler, self).__init__(save_dir)
        self._options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
        )

    def start(self):
        tf.profiler.experimental.start(self._save_dir, options=self._options)

    def stop(self):
        tf.profiler.experimental.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
