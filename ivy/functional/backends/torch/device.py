"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature."""

# global
import os
import importlib
import torch
from typing import Optional, Union
from torch.profiler import ProfilerActivity
from torch.profiler import profile

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler

torch_scatter = None

# API #
# ----#


def dev(
    x: torch.Tensor, /, *, as_native: bool = False
) -> Union[ivy.Device, torch.device]:
    dv = x.device
    if as_native:
        return dv
    return as_ivy_dev(dv)


def to_device(
    x: torch.Tensor,
    device: Union[torch.device, ivy.Device],
    /,
    *,
    stream: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is not None:
        current_dev = dev(x, as_native=True)
        native_dev = as_native_dev(device)
        if current_dev != native_dev:
            ret = x.to(device=native_dev)
            if isinstance(x, torch.nn.Parameter):
                return torch.nn.Parameter(ret)
            return ret
    return x


def as_ivy_dev(device, /):
    if isinstance(device, str):
        return ivy.Device(device)
    if is_native_dev(device):
        dev_type, dev_idx = (device.type, device.index)
        if dev_type == "cpu":
            return ivy.Device(dev_type)
        return ivy.Device(
            dev_type.replace("cuda", "gpu")
            + (":" + (str(dev_idx) if dev_idx is not None else "0"))
        )
    raise ivy.utils.exceptions.IvyError(
        f"{device} couldn't be converted to ivy device. "
        "Expcted a torch.device or a string of value type '(cpu|cuda|gpu)[:<index>]"
    )


def as_native_dev(
    device: Union[ivy.Device, str, torch.device],
    /,
) -> torch.device:
    if is_native_dev(device):
        return device  # type: ignore
    if isinstance(device, str):
        device = device.lower().replace("gpu", "cuda")
        return torch.device(device)
    else:
        raise ivy.utils.exceptions.IvyError(
            f"{device} couldn't be converted to torch.device. "
            "Expcted a torch.device or a valid torch device string or ivy.Device."
        )


def is_native_dev(device, /):
    return isinstance(device, torch.device)


def clear_cached_mem_on_dev(device: Union[ivy.Device, torch.device], /) -> None:
    torch_dev = as_native_dev(device)
    if torch_dev.type == "cuda":
        torch.cuda.empty_cache()


def num_gpus() -> int:
    return torch.cuda.device_count()


def gpu_is_available() -> bool:
    return torch.cuda.is_available()


# noinspection PyUnresolvedReferences
def tpu_is_available() -> bool:
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


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
