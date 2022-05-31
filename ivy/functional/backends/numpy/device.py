"""Collection of Numpy general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import time
import numpy as np
from typing import Union

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(x: np.ndarray, as_native: bool = False) -> Union[ivy.Device, str]:
    if as_native:
        return "cpu"
    return as_ivy_dev("cpu")


_dev_callable = dev


def as_ivy_dev(device):
    return ivy.Device("cpu")


def as_native_dev(device):
    return "cpu"


def clear_mem_on_dev(device):
    return None


def tpu_is_available() -> bool:
    return False


def num_gpus() -> int:
    return 0


def gpu_is_available() -> bool:
    return False


# private version of to_dev to be used in backend implementations
def _to_dev(x: np.ndarray, device=None) -> np.ndarray:
    """private version of `to_dev` to be used in backend implementations"""
    if device is not None:
        if "gpu" in device:
            raise Exception(
                "Native Numpy does not support GPU placement, "
                "consider using Jax instead"
            )
        elif "cpu" in device:
            pass
        else:
            raise Exception(
                "Invalid device specified, must be in the form "
                "[ 'cpu:idx' | 'gpu:idx' ], but found {}".format(device)
            )
    return x


def to_dev(x: np.ndarray, *, device: str) -> np.ndarray:
    if device is not None:
        if "gpu" in device:
            raise Exception(
                "Native Numpy does not support GPU placement, "
                "consider using Jax instead"
            )
        elif "cpu" in device:
            pass
        else:
            raise Exception(
                "Invalid device specified, must be in the form "
                "[ 'cpu:idx' | 'gpu:idx' ], but found {}".format(device)
            )
    return x


class Profiler(BaseProfiler):
    def __init__(self, save_dir):
        # ToDO: add proper numpy profiler
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
