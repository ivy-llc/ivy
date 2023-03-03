# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def adjoint(input):
    return ivy.adjoint(input)


@to_ivy_arrays_and_back
def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


@to_ivy_arrays_and_back
def chunk(input, chunks, dim=0):
    if ivy.shape(input) == ():
        return [input]
    else:
        dim_size = ivy.shape(input)[dim]
        chunk_size = dim_size // chunks
        if chunk_size == 0:
            return ivy.split(input, num_or_size_splits=dim_size, axis=dim)
        else:
            remainder = dim_size % chunks
            if remainder == 0:
                return ivy.split(input, num_or_size_splits=chunks, axis=dim)
            else:
                return ivy.split(
                    input,
                    num_or_size_splits=tuple(
                        [chunk_size + remainder] + [chunk_size] * (chunks - 1)
                    ),
                    axis=dim,
                )


@to_ivy_arrays_and_back
def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


@to_ivy_arrays_and_back
def gather(input, dim, index, *, sparse_grad=False, out=None):
    if sparse_grad:
        raise ivy.utils.exceptions.IvyException(
            "Gather does not yet support the sparse grad functionality"
        )

    dim = dim % len(input.shape)
    all_indices = ivy.argwhere(ivy.full(index.shape, True))
    gather_locations = ivy.reshape(index, [ivy.prod(ivy.array(index.shape))])

    gather_indices = []
    for axis in range(len(index.shape)):
        if axis == dim:
            gather_indices.append(ivy.array(gather_locations, dtype=index.dtype))
        else:
            gather_indices.append(ivy.array(all_indices[:, axis], dtype=index.dtype))

    gather_indices = ivy.stack(gather_indices, axis=-1)
    gathered = ivy.gather_nd(input, gather_indices)
    reshaped = ivy.reshape(gathered, index.shape)
    return reshaped


@to_ivy_arrays_and_back
def nonzero(input, *, out=None, as_tuple=False):
    ret = ivy.nonzero(input)
    if as_tuple is False:
        ret = ivy.matrix_transpose(ivy.stack(ret))

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_ivy_arrays_and_back
def permute(input, dims):
    return ivy.permute_dims(input, axes=dims)


@to_ivy_arrays_and_back
def reshape(input, shape):
    return ivy.reshape(input, shape)


@to_ivy_arrays_and_back
def as_strided(input, size, stride, storage_offset=None):
    ind = ivy.array([0], dtype=ivy.int64)
    for i, (size_i, stride_i) in enumerate(zip(size, stride)):
        r_size = [1] * len(stride)
        r_size[i] = -1
        ind = ind + ivy.reshape(ivy.arange(size_i), r_size) * stride_i
    if storage_offset:
        ind = ind + storage_offset
    return ivy.gather(ivy.flatten(input), ind)


@to_ivy_arrays_and_back
def squeeze(input, dim):
    if isinstance(dim, int) and input.ndim > 0:
        if input.shape[dim] > 1:
            return input
    return ivy.squeeze(input, dim)


@to_ivy_arrays_and_back
def stack(tensors, dim=0, *, out=None):
    return ivy.stack(tensors, axis=dim, out=out)


@to_ivy_arrays_and_back
def swapaxes(input, axis0, axis1):
    return ivy.swapaxes(input, axis0, axis1)


@to_ivy_arrays_and_back
def swapdims(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


@to_ivy_arrays_and_back
def transpose(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


@to_ivy_arrays_and_back
def t(input):
    if input.ndim > 2:
        raise ivy.utils.exceptions.IvyException(
            "t(input) expects a tensor with <= 2 dimensions, but self is %dD"
            % input.ndim
        )
    if input.ndim == 2:
        return ivy.swapaxes(input, 0, 1)
    else:
        return input


@to_ivy_arrays_and_back
def tile(input, dims):
    try:
        tup = tuple(dims)
    except TypeError:
        tup = (dims,)
    d = len(tup)
    res = 0
    if len(input.shape) > len([dims]) - 1:
        res = input
    if d < input.ndim:
        tup = (1,) * (input.ndim - d) + tup
        res = ivy.tile(input, tup)

    else:
        res = ivy.tile(input, repeats=dims, out=None)
    return res


@to_ivy_arrays_and_back
def unsqueeze(input, dim=0):
    return ivy.expand_dims(input, axis=dim)


@to_ivy_arrays_and_back
def argwhere(input):
    return ivy.argwhere(input)


@to_ivy_arrays_and_back
def movedim(input, source, destination):
    return ivy.moveaxis(input, source, destination)


@to_ivy_arrays_and_back
def hstack(tensors, *, out=None):
    return ivy.hstack(tensors, out=out)


@to_ivy_arrays_and_back
def index_select(input, dim, index, *, out=None):
    return ivy.gather(input, index, axis=dim, out=out)


@to_ivy_arrays_and_back
def dstack(tensors, *, out=None):
    return ivy.dstack(tensors, out=out)


@to_ivy_arrays_and_back
def take_along_dim(input, indices, dim, *, out=None):
    return ivy.take_along_axis(input, indices, dim, out=out)


@to_ivy_arrays_and_back
def vstack(tensors, *, out=None):
    return ivy.vstack(tensors, out=out)


@to_ivy_arrays_and_back
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        split_size_or_sections = [split_size] * (tensor.shape[dim] // split_size)
        if tensor.shape[dim] % split_size:
            split_size_or_sections.append(tensor.shape[dim] % split_size)
    return tuple(
        ivy.split(
            tensor,
            num_or_size_splits=split_size_or_sections,
            axis=dim,
            with_remainder=True,
        )
    )


@to_ivy_arrays_and_back
def tensor_split(input, indices_or_sections, dim=0):
    if isinstance(indices_or_sections, (list, tuple)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[dim]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.split(
        input, num_or_size_splits=indices_or_sections, axis=dim, with_remainder=False
    )


@to_ivy_arrays_and_back
def unbind(input, dim=0):
    shape = list(input.shape)
    shape.pop(dim)
    return tuple([x.reshape(tuple(shape)) for x in split(input, 1, dim=dim)])


def _get_indices_or_sections(indices_or_sections, indices, sections):
    if not ivy.exists(indices_or_sections):
        if ivy.exists(indices) and not ivy.exists(sections):
            indices_or_sections = indices
        elif ivy.exists(sections) and not ivy.exists(indices):
            indices_or_sections = sections
        else:
            raise ivy.utils.exception.IvyError(
                "got invalid argument for indices_or_sections"
            )
    return indices_or_sections


@to_ivy_arrays_and_back
def dsplit(input, indices_or_sections=None, /, *, indices=None, sections=None):
    indices_or_sections = _get_indices_or_sections(
        indices_or_sections, indices, sections
    )
    return tuple(ivy.dsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def hsplit(input, indices_or_sections=None, /, *, indices=None, sections=None):
    indices_or_sections = _get_indices_or_sections(
        indices_or_sections, indices, sections
    )
    return tuple(ivy.hsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def vsplit(input, indices_or_sections=None, /, *, indices=None, sections=None):
    indices_or_sections = _get_indices_or_sections(
        indices_or_sections, indices, sections
    )
    return tuple(ivy.vsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def row_stack(tensors, *, out=None):
    return ivy.vstack(tensors, out=out)
