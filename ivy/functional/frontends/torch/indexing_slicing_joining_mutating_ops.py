# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


def chunk(input, chunks, dim=0):
    return ivy.split(input, num_or_size_splits=chunks, axis=dim, with_remainder=True)


def nonzero(input, *, out=None, as_tuple=False):
    ret = ivy.nonzero(input)
    if as_tuple is False:
        ret = ivy.matrix_transpose(ivy.stack(ret))

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def permute(input, dims):
    return ivy.permute_dims(input, axes=dims)


def reshape(input, shape):
    return ivy.reshape(input, shape)


def swapdims(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


def swapaxes(input, axis0, axis1):
    return ivy.swapaxes(input, axis0, axis1)


def transpose(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


def stack(tensors, dim=0, *, out=None):
    return ivy.stack(tensors, axis=dim, out=out)


def squeeze(input, dim):
    if isinstance(dim, int):
        if input.shape[dim] > 1:
            return input if ivy.is_ivy_array(input) else ivy.array(input)
    return ivy.squeeze(input, dim)
