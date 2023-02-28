"""Collection of Mindspore general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import importlib
import mindspore as ms
from typing import Optional, Union

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler

# API #
# ----#


def dev(x: ms.Tensor, /, *, as_native: bool = False) -> Union[ivy.Device, str]:
    if as_native:
        return "cpu"
    return as_ivy_dev("cpu")


def dev(x: ms.Tensor, /, *, as_native: bool = False) -> Union[ivy.Device, str]:
    if as_native:
        return "cpu"
    return as_ivy_dev("cpu")


def as_ivy_dev(device: str, /):
    return ivy.Device("cpu")


def as_native_dev(device: str, /):
    return "cpu"


def clear_mem_on_dev(device: str, /):
    return None


def tpu_is_available() -> bool:
    return False


def num_gpus() -> int:
    return 0


def gpu_is_available() -> bool:
    return False


# private version of to_device to be used in backend implementations
def _to_device(x: ms.Tensor, device=None) -> ms.Tensor:
    """Private version of `to_device` to be used in backend implementations"""
    if device is not None:
        if "gpu" in device:
            raise ivy.utils.exceptions.IvyException(
                "GPU is currently not supported for mindspore"
            )
        elif "cpu" in device:
            pass
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid device specified, must be in the form "
                "[ 'cpu:idx' | 'gpu:idx' ], but found {}".format(device)
            )
    return x


