"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import os
import torch as _torch
import importlib
torch_scatter = None
from typing import Optional
from torch.profiler import ProfilerActivity
from torch.profiler import profile as _profile

# local
from ivy.core.device import Profiler as BaseProfiler


# API #
# ----#

def dev(x):
    return x.device


def to_dev(x, dev_str: Optional[str] = None):
    ret = x.to(str_to_dev(dev_str))
    if isinstance(x, _torch.nn.Parameter):
        return _torch.nn.Parameter(ret)
    return ret


def dev_to_str(dev_in: _torch.device):
    dev_type, dev_idx = (dev_in.type, dev_in.index)
    if dev_type == 'cpu':
        return dev_type
    return dev_type.replace('cuda', 'gpu') + (':' + (str(dev_idx) if dev_idx is not None else '0'))


def str_to_dev(dev_str: Optional[str] = None) -> Optional[_torch.device]:
    if dev_str is None:
        return dev_str
    return _torch.device(dev_str.replace('gpu', 'cuda'))


def dev_str(x):
    return dev_to_str(dev(x))


_callable_dev_str = dev_str
gpu_is_available = _torch.cuda.is_available
num_gpus = _torch.cuda.device_count


# noinspection PyUnresolvedReferences
def tpu_is_available():
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


class Profiler(BaseProfiler):

    def __init__(self, save_dir):
        super(Profiler, self).__init__(save_dir)
        self._prof = _profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)

    def start(self):
        self._prof.__enter__()

    def stop(self):
        self._prof.__exit__(None, None, None)
        self._prof.export_chrome_trace(os.path.join(self._save_dir, 'trace.json'))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
