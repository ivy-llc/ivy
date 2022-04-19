"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import os

_round = round
import mxnet as mx
from mxnet import profiler as _profiler

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler


def dev(x, as_str=False):
    dv = x.context
    if as_str:
        return dev_to_str(dv)
    return dv


def to_dev(x, dev=None, out=None):
    if dev is not None:
        ret = x.as_in_context(dev_from_str(dev))
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret 
    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x


def dev_to_str(dev):
    if isinstance(dev, str):
        return dev
    device_type = dev.device_type
    if device_type == 'cpu':
        return device_type
    return device_type + (':' + (str(dev.device_id) if dev.device_id is not None else '0'))


def dev_from_str(dev):
    if not isinstance(dev, str):
        return dev
    dev_split = dev.split(':')
    dev = dev_split[0]
    if len(dev_split) > 1:
        idx = int(dev_split[1])
    else:
        idx = 0
    return mx.context.Context(dev, idx)


def gpu_is_available() -> bool:
    return mx.context.num_gpus() > 0


clear_mem_on_dev = lambda dev: None
_callable_dev = dev
num_gpus = lambda: mx.context.num_gpus()
tpu_is_available = lambda: False


class Profiler(BaseProfiler):

    def __init__(self, save_dir):
        super(Profiler, self).__init__(save_dir)
        self._prof = _profiler
        self._prof.set_config(profile_all=True,
                              aggregate_stats=True,
                              continuous_dump=True,
                              filename=os.path.join(save_dir, 'trace.json'))

    def start(self):
        self._prof.set_state('run')

    def stop(self):
        self._prof.set_state('stop')
        self._prof.dump()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
