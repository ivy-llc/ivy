"""
Tensorflow device functions.

Collection of TensorFlow general functions, wrapped to fit Ivy syntax
and signature.
"""

# global
_round = round
import tensorflow as tf
from typing import Union, Optional

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    as_native: bool = False,
) -> Union[ivy.Device, str]:
    dv = x.device
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_device(
    x: Union[tf.Tensor, tf.Variable],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if device is not None:
        native_device = as_native_dev(device)
        current_dev = dev(x, as_native=True)
        if not _same_device(current_dev, native_device):
            with tf.device(native_device):
                return tf.identity(x)
    return x


def as_ivy_dev(device: str, /):
    if isinstance(device, str) and "/" not in device:
        return ivy.Device(device)
    if is_native_dev(device):
        dev_in_split = device[1:].split(":")[-2:]
        dev_type, dev_idx = dev_in_split
        dev_type = dev_type.lower()
        if dev_type == "cpu" and dev_idx == "0":
            return ivy.Device("cpu")
        return ivy.Device(":".join([dev_type, dev_idx]))
    else:
        raise ivy.utils.exceptions.IvyException(
            f"Cannot convert {device} to an ivy device. Expected a "
            f"str, got {type(device)}"
        )


def as_native_dev(device: str, /):
    if is_native_dev(device):
        return _shorten_device(device)
    if isinstance(device, str) and "/" not in device:
        ret = "/" + ivy.Device(device).upper()
        if not ret[-1].isnumeric():
            ret += ":0"
        return ret
    else:
        raise ivy.utils.exceptions.IvyError(
            f"Cannot convert {device} to an ivy device. Expected a "
            f"str, got {type(device)}"
        )


def is_native_dev(device: str, /):
    if isinstance(device, str) and device[0] == "/":
        dev_in_split = device[1:].split(":")[-2:]
        if len(dev_in_split) == 2:
            if dev_in_split[0] in ["CPU", "GPU", "TPU"] and dev_in_split[1].isnumeric():
                return True
    return False


def _shorten_device(device: str):
    return "/" + ":".join(device[1:].split(":")[-2:])


def _same_device(dev_a, dev_b):
    if dev_a is None or dev_b is None:
        return False
    return _shorten_device(dev_a) == _shorten_device(dev_b)


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
