"""
Collection of Jax device functions, wrapped to fit Ivy syntax and signature.
"""

# global
import os
import jax as _jax

# local
from ivy.core.device import Profiler as BaseProfiler


# Helpers #
# --------#

def _to_array(x):
    if isinstance(x, _jax.interpreters.ad.JVPTracer):
        return _to_array(x.primal)
    elif isinstance(x, _jax.interpreters.partial_eval.DynamicJaxprTracer):
        return _to_array(x.aval)
    return x


# API #
# ----#

def dev(x):
    if isinstance(x, _jax.interpreters.partial_eval.DynamicJaxprTracer):
        return None
    return _to_array(x).device_buffer.device()


def to_dev(x, dev_str=None):
    if dev_str is not None:
        cur_dev_str = dev_to_str(dev(x))
        if cur_dev_str != dev_str:
            x = _jax.device_put(x, str_to_dev(dev_str))
    return x


def dev_to_str(dev_in):
    if dev_in is None:
        return None
    p, dev_id = (dev_in.platform, dev_in.id)
    if p == 'cpu':
        return p
    return p + ':' + str(dev_id)


def str_to_dev(dev_str):
    dev_split = dev_str.split(':')
    dev_str = dev_split[0]
    if len(dev_split) > 1:
        idx = int(dev_split[1])
    else:
        idx = 0
    return _jax.devices(dev_str)[idx]


dev_str = lambda x: dev_to_str(dev(x))
dev_str.__name__ = 'dev_str'


def _dev_is_available(base_dev_str):
    try:
        _jax.devices(base_dev_str)
        return True
    except RuntimeError:
        return False


gpu_is_available = lambda: _dev_is_available('gpu')


def num_gpus():
    try:
        return len(_jax.devices('gpu'))
    except RuntimeError:
        return 0


tpu_is_available = lambda: _dev_is_available('tpu')


# noinspection PyMethodMayBeStatic
class Profiler(BaseProfiler):

    def __init__(self, save_dir):
        super(Profiler, self).__init__(save_dir)
        self._save_dir = os.path.join(self._save_dir, 'profile')

    def start(self):
        _jax.profiler.start_trace(self._save_dir)

    def stop(self):
        _jax.profiler.stop_trace()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
