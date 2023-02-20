"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import paddle
from typing import Optional, Union
import time

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from paddle.fluid.libpaddle import Place
from ivy.exceptions import IvyNotImplementedException


# API #
# ----#


def dev(
    x: paddle.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, Place]:
    raise IvyNotImplementedException()


def to_device(
    x: paddle.Tensor,
    device: Place,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def as_ivy_dev(device: Place, /):
    raise IvyNotImplementedException()


def as_native_dev(
    device: Optional[Union[ivy.Device, Place]] = None,
    /,
) -> Optional[Place]:
    raise IvyNotImplementedException()


def clear_mem_on_dev(device: Place, /):
    raise IvyNotImplementedException()


def num_gpus() -> int:
    raise IvyNotImplementedException()


def gpu_is_available() -> bool:
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    raise IvyNotImplementedException()


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        # ToDO: add proper Paddle profiler
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
