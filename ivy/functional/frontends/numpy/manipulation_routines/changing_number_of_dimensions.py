# global
import ivy


# squeeze
def squeeze(
    a,
    axis=None,
):
    return ivy.squeeze(a, axis)


# expand_dims
def expand_dims(
    a,
    axis,
):
    return ivy.expand_dims(a, axis=axis)
