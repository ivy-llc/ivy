"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import os
_round = round
import mxnet as _mx
from mxnet import profiler as _profiler

# local
from ivy.core.device import Profiler as BaseProfiler


dev = lambda x: x.context
dev.__name__ = 'dev'


def to_dev(x, dev_str=None):
    if dev_str is not None:
        return x.as_in_context(str_to_dev(dev_str))
    return x


def dev_to_str(dev_in):
    device_type = dev_in.device_type
    if device_type == 'cpu':
        return device_type
    return device_type + (':' + (str(dev_in.device_id) if dev_in.device_id is not None else '0'))


def str_to_dev(dev_str):
    dev_split = dev_str.split(':')
    dev_str = dev_split[0]
    if len(dev_split) > 1:
        idx = int(dev_split[1])
    else:
        idx = 0
    return _mx.context.Context(dev_str, idx)


dev_str = lambda x: dev_to_str(dev(x))
dev_str.__name__ = 'dev_str'
_callable_dev_str = dev_str
gpu_is_available = lambda: _mx.context.num_gpus() > 0
num_gpus = lambda: _mx.context.num_gpus()
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
