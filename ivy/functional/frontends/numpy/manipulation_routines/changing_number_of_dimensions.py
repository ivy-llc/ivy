# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


# atleast_1d
@to_ivy_arrays_and_back
def atleast_1d(
    *arys,
):
    return ivy.atleast_1d(*arys)


@to_ivy_arrays_and_back
def atleast_2d(*arys):
    return ivy.atleast_2d(*arys)


@to_ivy_arrays_and_back
def atleast_3d(*arys):
    return ivy.atleast_3d(*arys)


# broadcast_arrays
@to_ivy_arrays_and_back
def broadcast_arrays(*args):
    return ivy.broadcast_arrays(*args)


# expand_dims
@to_ivy_arrays_and_back
def expand_dims(
    a,
    axis,
):
    return ivy.expand_dims(a, axis=axis)


# squeeze
@to_ivy_arrays_and_back
def squeeze(
    a,
    axis=None,
):
    return ivy.squeeze(a, axis=axis)
