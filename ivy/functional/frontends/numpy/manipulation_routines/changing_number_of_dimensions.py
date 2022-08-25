# global
import ivy


# squeeze
def squeeze(
    x,
    axis,
    /,
):
    ret = ivy.squeeze(x, axis)
    return ret


squeeze.unsupported_dtypes = {"torch": ("float16",)}
