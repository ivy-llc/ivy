"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
_round = round
import tensorflow as _tf

# local
from ivy.core.device import Profiler as BaseProfiler


def _same_device(dev_a, dev_b):
    return '/' + ':'.join(dev_a[1:].split(':')[-2:]) == '/' + ':'.join(dev_b[1:].split(':')[-2:])


dev = lambda x: x.device


def to_dev(x, dev_str=None):
    current_dev_str = _dev_str_callable(x)
    if not _same_device(current_dev_str, dev_str):
        with _tf.device('/' + dev_str.upper()):
            return _tf.identity(x)
    return x


def dev_to_str(dev_in):
    return ':'.join(dev_in[1:].split(':')[-2:]).lower()


def str_to_dev(dev_str):
    return '/' + dev_str.upper()


dev_str = lambda x: dev_to_str(dev(x))
_dev_str_callable = dev_str
gpu_is_available = lambda: len(_tf.config.list_physical_devices('GPU')) > 0
num_gpus = lambda: len(_tf.config.list_physical_devices('GPU'))


def tpu_is_available():
    try:
        resolver = _tf.distribute.cluster_resolver.TPUClusterResolver()
        _tf.config.experimental_connect_to_cluster(resolver)
        _tf.tpu.experimental.initialize_tpu_system(resolver)
        _tf.config.list_logical_devices('TPU')
        _tf.distribute.experimental.TPUStrategy(resolver)
        return True
    except ValueError:
        return False


class Profiler(BaseProfiler):

    def __init__(self, save_dir):
        super(Profiler, self).__init__(save_dir)
        self._options = _tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=1, device_tracer_level=1)

    def start(self):
        _tf.profiler.experimental.start(self._save_dir, options=self._options)

    def stop(self):
        _tf.profiler.experimental.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
