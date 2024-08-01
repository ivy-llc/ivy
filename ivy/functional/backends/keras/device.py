"""Tensorflow device functions.

Collection of TensorFlow general functions, wrapped to fit Ivy syntax
and signature.
"""

# global
_round = round
import tensorflow as tf
from typing import Union, Optional

# local
import ivy
from ivy.functional.ivy.device import Profiler as BaseProfiler
from .func_wrapper import use_keras_backend_framework


@use_keras_backend_framework
def dev(x, /, *, as_native=False):
    pass


@use_keras_backend_framework
def to_device(x, device, /, *, stream=None, out=None):
    pass


@use_keras_backend_framework
def as_ivy_dev(device, /):
    pass


@use_keras_backend_framework
def as_native_dev(device, /):
    pass


@use_keras_backend_framework
def clear_cached_mem_on_dev(device, /):
    pass


@use_keras_backend_framework
def num_gpus():
    pass


@use_keras_backend_framework
def gpu_is_available():
    pass


@use_keras_backend_framework
def tpu_is_available():
    pass


@use_keras_backend_framework
def handle_soft_device_variable(*args, fn, **kwargs):
    pass
