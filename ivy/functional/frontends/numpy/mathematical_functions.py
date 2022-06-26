# global
import ivy


def tan(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None,
        subok=True):
    if dtype:
        x = ivy.astype(x, ivy.as_ivy_dtype(dtype))
    ret = ivy.tan(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, x, out=out)
    return ret
<<<<<<< Updated upstream

tan.unsupported_dtypes = {"torch": ("float16",)}
=======
>>>>>>> Stashed changes
