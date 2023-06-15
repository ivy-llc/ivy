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
def squeeze(input, dim=None):
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
def moveaxis(input, source, destination):
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
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[dim]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.split(
        input, num_or_size_splits=indices_or_sections, axis=dim, with_remainder=True
    )


@to_ivy_arrays_and_back
def unbind(input, dim=0):
    shape = list(input.shape)
    shape.pop(dim)
    return tuple([x.reshape(tuple(shape)) for x in split(input, 1, dim=dim)])


@to_ivy_arrays_and_back
def dsplit(input, indices_or_sections, /):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[2]])
            .astype(ivy.int8)
            .to_list()
        )
    return tuple(ivy.dsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def hsplit(input, indices_or_sections=None, /):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        if input.ndim == 1:
            indices_or_sections = (
                ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[0]])
                .astype(ivy.int8)
                .to_list()
            )
        else:
            indices_or_sections = (
                ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[1]])
                .astype(ivy.int8)
                .to_list()
            )
    return tuple(ivy.hsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def vsplit(input, indices_or_sections=None, /):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[input.shape[0]])
            .astype(ivy.int8)
            .to_list()
        )
    return tuple(ivy.vsplit(input, indices_or_sections))


@to_ivy_arrays_and_back
def row_stack(tensors, *, out=None):
    return ivy.vstack(tensors, out=out)


@to_ivy_arrays_and_back
def where(condition, input=None, other=None):
    if not ivy.exists(input) and not ivy.exists(other):
        return nonzero(condition, as_tuple=True)
    return ivy.where(condition, input, other)


@to_ivy_arrays_and_back
def conj(input):
    return ivy.conj(input)


@to_ivy_arrays_and_back
def index_add(input, dim, index, source, *, alpha=1, out=None):
    input = ivy.swapaxes(input, dim, 0)
    source = ivy.swapaxes(source, dim, 0)
    _to_adds = []
    index = sorted(zip(ivy.to_list(index), range(len(index))), key=(lambda x: x[0]))
    while index:
        _curr_idx = index[0][0]
        while len(_to_adds) < _curr_idx:
            _to_adds.append(ivy.zeros_like(source[0]))
        _to_add_cum = ivy.get_item(source, index[0][1])
        while (1 < len(index)) and (index[0][0] == index[1][0]):
            _to_add_cum = _to_add_cum + ivy.get_item(source, index.pop(1)[1])
        index.pop(0)
        _to_adds.append(_to_add_cum)
    while len(_to_adds) < input.shape[0]:
        _to_adds.append(ivy.zeros_like(source[0]))
    _to_adds = ivy.stack(_to_adds)
    if len(input.shape) < 2:
        # Added this line due to the paddle backend treating scalars as 1-d arrays
        _to_adds = ivy.flatten(_to_adds)

    ret = ivy.add(input, _to_adds, alpha=alpha)
    ret = ivy.swapaxes(ret, 0, dim, out=out)
    return ret


@to_ivy_arrays_and_back
def index_copy(input, dim, index, source, *, out=None):
    input = ivy.swapaxes(input, dim, 0)
    source = ivy.swapaxes(source, dim, 0)
    index = sorted(zip(ivy.to_list(index), range(len(index))), key=(lambda x: x[0]))
    res = []
    while index:
        _curr_idx = index[0][0]
        for i in range(len(res), _curr_idx):
            res.append(ivy.get_item(input, i))
        while (1 < len(index)) and (index[0][0] == index[1][0]):
            index.pop(0)
        res.append(ivy.get_item(source, index[0][1]))
        index.pop(0)
    for i in range(len(res), input.shape[0]):
        res.append(ivy.get_item(input, i))
    res = ivy.stack(res)
    if len(input.shape) < 2:
        res = ivy.flatten(res)

    return ivy.swapaxes(res, 0, dim, out=out)


@to_ivy_arrays_and_back
def masked_select(input, mask, out=None):
    return ivy.flatten(input[mask], out=out)


@to_ivy_arrays_and_back
def take(input, index):
    input = ivy.reshape(input, (-1,))
    return ivy.gather(input, index, axis=0)


@to_ivy_arrays_and_back
def narrow(input, dim, start, length):
    num_dims = ivy.get_num_dims(input)
    slices = [slice(None)] * num_dims
    slices[dim] = slice(start, start + length)
    return input[tuple(slices)]
