# global
import keras
from typing import Union, Optional

# local
import ivy
from . import backend_version
from ivy.functional.backends.jax import JaxArray, NativeArray
from .func_wrapper import use_keras_backend_framework


@use_keras_backend_framework
def is_native_array(x, /, *, exclusive=False):
    pass


@use_keras_backend_framework
def array_equal(x0, x1, /):
    pass


def container_types():
    return []


def current_backend_str() -> str:
    return "keras"


@use_keras_backend_framework
def get_item(x, /, query, *, copy: Optional[bool] = None):
    pass


@use_keras_backend_framework
def to_numpy(x, /, *, copy=True):
    pass


@use_keras_backend_framework
def to_scalar(x, /):
    pass


@use_keras_backend_framework
def to_list(x, /):
    pass


@use_keras_backend_framework
def gather(params, indices, /, *, axis=-1, batch_dims=0, out=None):
    pass


@use_keras_backend_framework
def gather_nd(params, indices, /, *, batch_dims=0, out=None):
    pass


@use_keras_backend_framework
def get_num_dims(x, /, *, as_array=False):
    pass


@use_keras_backend_framework
def size(x, /):
    pass


@use_keras_backend_framework
def inplace_arrays_supported():
    pass


@use_keras_backend_framework
def inplace_decrement(x, val):
    pass


@use_keras_backend_framework
def inplace_increment(x, val):
    pass


@use_keras_backend_framework
def inplace_update(x, val, /, *, ensure_in_backend=False, keep_input_dtype=False):
    pass


@use_keras_backend_framework
def inplace_variables_supported(): 
    pass


@use_keras_backend_framework
def multiprocessing(context=None):
    pass


@use_keras_backend_framework
def scatter_flat(
    indices,
    updates,
    /,
    *,
    size=None,
    reduction="sum",
    out=None,
):
    pass


@use_keras_backend_framework
def scatter_nd(
    indices,
    updates,
    /,
    shape=None,
    *,
    reduction="sum",
    out=None,
):
    pass


@use_keras_backend_framework
def shape(x, /, *, as_array=False):
    pass


@use_keras_backend_framework
def vmap(func, in_axes=0, out_axes=0):
    pass


@use_keras_backend_framework
def isin(elements, test_elements, /, *, assume_unique=False, invert=False):
    pass


@use_keras_backend_framework
def itemsize(x):
    pass
