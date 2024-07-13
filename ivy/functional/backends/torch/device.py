"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and
signature."""

import inspect

# global
import os
import importlib
import torch
from typing import Optional, Union
from torch.profiler import ProfilerActivity
from torch.profiler import profile

# local
import ivy
from ivy.functional.ivy.device import (
    _shift_native_arrays_on_default_device,
    Profiler as BaseProfiler,
)

torch_scatter = None

# API #
# ----#


def dev(
    x: torch.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, torch.device]:
    dv = x.device
    if as_native:
        if isinstance(dv, torch.device):
            dv = dv.type
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device(dv.replace("gpu", "mps"))
        return torch.device(dv.replace("gpu", "cuda"))
    return as_ivy_dev(dv)


def to_device(
    x: torch.Tensor,
    device: torch.device,
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is None:
        return x
    ret = x.to(as_native_dev(device))
    if isinstance(x, torch.nn.Parameter):
        return torch.nn.Parameter(ret)
    return ret


def as_ivy_dev(device: torch.device, /):
    if isinstance(device, str):
        return ivy.Device(device)
    dev_type, dev_idx = (device.type, device.index)
    if dev_type == "cpu":
        return ivy.Device(dev_type)
    elif dev_type == "mps":
        return ivy.Device(
            dev_type.replace("mps", "gpu")
            + (":" + (str(dev_idx) if dev_idx is not None else "0"))
        )
    return ivy.Device(
        dev_type.replace("cuda", "gpu")
        + (":" + (str(dev_idx) if dev_idx is not None else "0"))
    )


def as_native_dev(
    device: Optional[Union[ivy.Device, torch.device]] = None,
    /,
) -> Optional[torch.device]:
    if not isinstance(device, str):
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device(ivy.Device(device).replace("gpu", "mps"))
    return torch.device(ivy.Device(device).replace("gpu", "cuda"))


def clear_cached_mem_on_dev(device: Union[ivy.Device, torch.device], /) -> None:
    torch_dev = as_native_dev(device)
    if torch_dev.type == "cuda":
        torch.cuda.empty_cache()
    elif torch_dev.type == "mps":
        from torch import mps

        mps.empty_cache()


def num_gpus() -> int:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1
    return torch.cuda.device_count()


def gpu_is_available() -> bool:
    return (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ) or torch.cuda.is_available()


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


def handle_soft_device_variable(*args, fn, **kwargs):
    args, kwargs, device_shifting_dev = _shift_native_arrays_on_default_device(
        *args, **kwargs
    )
    # checking if this function accepts `device` argument
    # must be handled in the backend
    if "device" in inspect.signature(fn).parameters:
        kwargs["device"] = device_shifting_dev
    return fn(*args, **kwargs)


class Profiler(BaseProfiler):
    def __init__(self, save_dir: str):
        super().__init__(save_dir)
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        )

    def start(self):
        self._prof.__enter__()

    def stop(self):
        self._prof.__exit__(None, None, None)
        self._prof.export_chrome_trace(os.path.join(self._save_dir, "trace.json"))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
