import ivy    


def median_overwrite(a, axis=None, keepdims=False):
    a = ivy.array(a)
    if keepdims:
        dims = a.shape()

    if axis is not None:
        a[axis] = ivy.sort(a[axis])
    else:
        a = ivy.sort(a.reshape(1, -1))

    if len(a) % 2 == 0:
        med = a[len(a) / 2]
    else:
        med = (a[(len(a) - 1) / 2] + a[(len(a) + 1) / 2]) / 2

    if keepdims:
        ivy.reshape(a, dims)

    return med


def _median(a, axis=None):
    if axis is not None:
        b = ivy.sort(ivy.array(a[axis]))
    else:
        b = ivy.sort(ivy.array(a).reshape(1, -1))

    if len(b) % 2 == 0:
        med = b[len(b) / 2]
    else:
        med = (b[(len(b) - 1) / 2] + b[(len(b) + 1) / 2]) / 2

    return med


def median(a, *, axis=None, out=None, overwrite_input=False, keepdims=False):

    if overwrite_input:
        med = median_overwrite(a, axis, keepdims)
    else:
        med = _median(a, axis)

    if out is not None:
        out = med
        return

    return med
