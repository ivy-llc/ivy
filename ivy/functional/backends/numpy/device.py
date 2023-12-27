"""Collection of Numpy general functions, wrapped to fit Ivy syntax and
signature."""

# global
import os
import time
import logging
import numpy as np
from typing import Union, Optional, Any

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(x: np.ndarray, /, *, as_native: bool = False) -> Union[ivy.Device, str]:
    if as_native:
        return "cpu"
    return as_ivy_dev("cpu")


def as_ivy_dev(device: str, /):
    if "gpu" in device:
        logging.warning(
            "Native Numpy does not support GPU placement, consider using Jax instead"
        )
    return ivy.Device("cpu")


def as_native_dev(device: str, /):
    if "gpu" in device:
        logging.warning(
            "Native Numpy does not support GPU placement, consider using Jax instead"
        )
    return "cpu"


def clear_cached_mem_on_dev(device: str, /):
    return None


def tpu_is_available() -> bool:
    return False


def num_gpus() -> int:
    return 0


def gpu_is_available() -> bool:
    return False


def to_device(
    x: np.ndarray,
    device: str,
    /,
    *,
    stream: Optional[Union[int, Any]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if device is not None:
        device = as_native_dev(device)
    return x


def handle_soft_device_variable(*args, fn, **kwargs):
    return fn(*args, **kwargs)


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        # ToDO: add proper numpy profiler
        super().__init__(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        time_taken = time.perf_counter() - self._start_time
        with open(os.path.join(self._save_dir, "profile.log"), "w+") as f:
            f.write(f"took {time_taken} seconds to complete")

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
