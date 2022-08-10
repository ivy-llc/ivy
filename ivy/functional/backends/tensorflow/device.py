"""Collection of TensorFlow general functions, wrapped to fit Ivy syntax and
signature.
"""

# global
_round = round
import tensorflow as tf
from typing import Union, Optional

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


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
    return as_ivy_dev(dv)


def to_device(
        x: Union[tf.Tensor, tf.Variable],
        device: str,
        /,
        *,
        stream: Optional[int] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if device is None:
        return x
    device = as_native_dev(device)
    current_dev = dev(x)
    if not _same_device(current_dev, device):
        with tf.device("/" + device.upper()):
            return tf.identity(x)
    return x


def as_ivy_dev(device:str, /):
    if isinstance(device, str) and "/" not in device:
        return ivy.Device(device)
    dev_in_split = device[1:].split(":")[-2:]
    if len(dev_in_split) == 1:
        return ivy.Device(dev_in_split[0])
    dev_type, dev_idx = dev_in_split
    dev_type = dev_type.lower()
    if dev_type == "cpu":
        return ivy.Device(dev_type)
    return ivy.Device(":".join([dev_type, dev_idx]))


def as_native_dev(device:str, /):
    if isinstance(device, str) and "/" in device:
        return device
    ret = "/" + ivy.Device(device).upper()
    if not ret[-1].isnumeric():
        ret += ":0"
    return ret


def clear_mem_on_dev(device:str, /):
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
