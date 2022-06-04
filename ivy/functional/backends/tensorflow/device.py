"""Collection of TensorFlow general functions, wrapped to fit Ivy syntax and
signature."""

# global
_round = round
import tensorflow as tf
from tensorflow.python.types.core import Tensor
import ivy

from typing import Union

# local
from ivy.functional.ivy.device import Profiler as BaseProfiler


def _same_device(dev_a, dev_b):
    if dev_a is None or dev_b is None:
        return False
    return "/" + ":".join(dev_a[1:].split(":")[-2:]) == "/" + ":".join(
        dev_b[1:].split(":")[-2:]
    )


def dev(x, as_str=False):
    dv = x.device
    if as_str:
        return as_ivy_dev(dv)
    return dv


def to_dev(x: Tensor, device=None, out: Tensor = None) -> Tensor:
    if device is None:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    current_dev = _dev_callable(x)
    if not _same_device(current_dev, device):
        with tf.device("/" + device.upper()):
            if ivy.exists(out):
                return ivy.inplace_update(out, tf.identity(x))
            return tf.identity(x)

    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x


def as_ivy_dev(device: tf.device)\
                -> str:
    if isinstance(device, str) and "/" not in device:
        return device
    dev_in_split = device[1:].split(":")[-2:]
    if len(dev_in_split) == 1:
        return dev_in_split[0]
    dev_type, dev_idx = dev_in_split
    dev_type = dev_type.lower()
    if dev_type == "cpu":
        return dev_type
    return ":".join([dev_type, dev_idx])


def dev_from_str(device):
    if isinstance(device, str) and "/" in device:
        return device
    ret = "/" + device.upper()
    if not ret[-1].isnumeric():
        ret = ret + ":0"
    return ret


clear_mem_on_dev = lambda dev: None
_dev_callable = dev


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
    def __init__(self, save_dir):
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
