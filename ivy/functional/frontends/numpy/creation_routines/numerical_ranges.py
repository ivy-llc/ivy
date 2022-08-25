# global
import ivy


def arange(start, stop=None, step=1, dtype=None, *, like=None):
    return ivy.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    ret = ivy.linspace(start, stop, num, axis=axis, endpoint=endpoint, dtype=dtype)
    if retstep:
        if endpoint:
            num -= 1
        step = (stop - start) / num
        return ret, step
    return ret


linspace.unsupported_dtypes = {"torch": ("float16",)}


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if not endpoint:
        interval = (stop - start) / num
        stop -= interval
    return ivy.logspace(start, stop, num, base=base, axis=axis, dtype=dtype)


logspace.unsupported_dtypes = {"torch": ("float16",)}


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    # Todo: add sparse check
    ret = ivy.meshgrid(*xi, indexing=indexing)
    if copy:
        return ivy.copy_array(ret)
    return ret
