"\nTensorflow device functions.\n\nCollection of TensorFlow general functions, wrapped to fit Ivy syntax\nand signature.\n"
from typing import Union, Optional
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(
    x: Union[(None, tf.Variable)], /, *, as_native: bool = False
) -> Union[(ivy.Device, str)]:
    raise NotImplementedError("mxnet.dev Not Implemented")


def to_device(
    x: Union[(None, tf.Variable)],
    device: str,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.to_device Not Implemented")


def as_ivy_dev(device: str, /):
    raise NotImplementedError("mxnet.as_ivy_dev Not Implemented")


def as_native_dev(device: str, /):
    raise NotImplementedError("mxnet.as_native_dev Not Implemented")


def clear_cached_mem_on_dev(device: str, /):
    raise NotImplementedError("mxnet.clear_cached_mem_on_dev Not Implemented")


def num_gpus() -> int:
    raise NotImplementedError("mxnet.num_gpus Not Implemented")


def gpu_is_available() -> bool:
    raise NotImplementedError("mxnet.gpu_is_available Not Implemented")


def tpu_is_available() -> bool:
    raise NotImplementedError("mxnet.tpu_is_available Not Implemented")


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        raise NotImplementedError("mxnet.__init__ Not Implemented")

    def start(self):
        raise NotImplementedError("mxnet.start Not Implemented")

    def stop(self):
        raise NotImplementedError("mxnet.stop Not Implemented")

    def __enter__(self):
        raise NotImplementedError("mxnet.__enter__ Not Implemented")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("mxnet.__exit__ Not Implemented")
