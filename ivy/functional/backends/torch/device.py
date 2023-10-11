"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature."""
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
    return ivy.as_ivy_dev(dv)


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
    ret = x.to(ivy.as_native_dev(device))
    if isinstance(x, torch.nn.Parameter):
        return torch.nn.Parameter(ret)
    return ret


def get_native_device_platform_and_id(device, /):
    return (device.type.replace("mps", "gpu").replace("cuda", "gpu"), device.index)


def get_native_device(device_platform, device_id, /):
    if torch.backends.mps.is_available():
        device_platform = device_platform.replace("gpu", "mps")
    else:
        device_platform = device_platform.replace("gpu", "cuda")
    return torch.device(device_platform + ":" + str(device_id))


def clear_cached_mem_on_dev(device: Union[ivy.Device, torch.device], /) -> None:
    torch_dev = ivy.as_native_dev(device)
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
        super(Profiler, self).__init__(save_dir)
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
