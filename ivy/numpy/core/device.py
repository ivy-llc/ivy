"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import os
import time

# local
from ivy.core.device import Profiler as BaseProfiler


dev = lambda x: 'cpu'
dev.__name__ = 'dev'
to_dev = lambda x, dev_str=None: x
dev_str = lambda x: 'cpu'
dev_str.__name__ = 'dev_str'
_dev_str_callable = dev_str
dev_to_str = lambda dev_in: 'cpu'
str_to_dev = lambda dev_str: 'cpu'
gpu_is_available = lambda: False
num_gpus = lambda: 0
tpu_is_available = lambda: False


class Profiler(BaseProfiler):

    def __init__(self, save_dir):
        # ToDO: add proper numpy profiler
        super(Profiler, self).__init__(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        time_taken = time.perf_counter() - self._start_time
        with open(os.path.join(self._save_dir, 'profile.log'), 'w+') as f:
            f.write('took {} seconds to complete'.format(time_taken))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
