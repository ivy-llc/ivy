import ivy
from .func_wrapper import use_keras_backend_framework


# Array API Standard #
# -------------------#


@use_keras_backend_framework
def arange(
    start,
    /,
    stop=None,
    step=1,
    *,
    dtype=None,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def asarray(
    obj,
    /,
    *,
    copy=None,
    dtype=None,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def empty(shape, *, dtype, device=None, out=None):
    pass


@use_keras_backend_framework
def empty_like(x, /, *, dtype, device=None, out=None):
    pass


@use_keras_backend_framework
def eye(
    n_rows,
    n_cols=None,
    /,
    *,
    k=0,
    batch_shape=None,
    dtype,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def to_dlpack(x, /, *, out=None):
    pass


@use_keras_backend_framework
def from_dlpack(x, /, *, out=None):
    pass


@use_keras_backend_framework
def full(
    shape,
    fill_value,
    *,
    dtype=None,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def full_like(
    x,
    /,
    fill_value,
    *,
    dtype,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def linspace(
    start,
    stop,
    /,
    num,
    *,
    axis=None,
    endpoint=True,
    dtype,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def meshgrid(*arrays, sparse=False, indexing="xy", out=None):
    pass


@use_keras_backend_framework
def ones(shape, *, dtype, device=None, out=None):
    pass


@use_keras_backend_framework
def ones_like(x, /, *, dtype, device=None, out=None):
    pass


@use_keras_backend_framework
def tril(x, /, *, k=0, out=None):
    pass


@use_keras_backend_framework
def triu(x, /, *, k=0, out=None):
    pass


@use_keras_backend_framework
def zeros(shape, *, dtype, device=None, out=None):
    pass


@use_keras_backend_framework
def zeros_like(x, /, *, dtype, device=None, out=None):
    pass


# Extra #
# ------#


array = asarray


@use_keras_backend_framework
def copy_array(x, *, to_ivy_array=True, out=None):
    pass


@use_keras_backend_framework
def one_hot(
    indices,
    depth,
    /,
    *,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    device=None,
    out=None,
):
    pass


@use_keras_backend_framework
def frombuffer(buffer, dtype=float, count=-1, offset=0):
    pass


@use_keras_backend_framework
def triu_indices(n_rows, n_cols=None, k=0, /, *, device=None):
    pass
