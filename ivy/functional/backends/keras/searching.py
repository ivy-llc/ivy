import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from .func_wrapper import use_keras_backend_framework


# Array API Standard #
# ------------------ #


@use_keras_backend_framework
def argmax(
    x,
    /,
    *,
    axis=None,
    keepdims=False,
    dtype=None,
    select_last_index=False,
    out=None,
):
    pass


@use_keras_backend_framework
def argmin(
    x,
    /,
    *,
    axis=None,
    keepdims=False,
    dtype=None,
    select_last_index=False,
    out=None,
):
    pass


@use_keras_backend_framework
def nonzero(
    x,
    /,
    *,
    as_tuple=True,
    size=None,
    fill_value=0,
):
    pass


@use_keras_backend_framework
def where(
    condition,
    x1,
    x2,
    /,
    *,
    out=None,
):
    pass


# Extra #
# ----- #


@use_keras_backend_framework
def argwhere(
    x,
    /,
    *,
    out=None,
):
    pass
