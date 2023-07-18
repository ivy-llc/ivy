import ivy

from typing import Union, Sequence, Tuple, Optional

from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_device_shifting,
)

from ivy.utils.exceptions import handle_exceptions


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def unfold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    input
        input tensor to be unfolded
    mode
        indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ret
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return ivy.reshape(ivy.moveaxis(input, mode, 0), (input.shape[mode], -1), out=out)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def fold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: int,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Refolds the mode-`mode` unfolding into a tensor of shape `shape` In other words,
    refolds the n-mode unfolded tensor into the original tensor of the specified shape.

    Parameters
    ----------
    input
        unfolded tensor of shape ``(shape[mode], -1)``
    mode
        the mode of the unfolding
    shape
        shape of the original tensor before unfolding

    Returns
    -------
    ret
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return ivy.moveaxis(ivy.reshape(input, full_shape), 0, mode, out=out)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_unfold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: Optional[int] = 0,
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    ravel_tensors: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
):
    """Partial unfolding of a tensor while ignoring the specified number
        of dimensions at the beginning and the end.
        For instance, if the first dimension of the tensor is the number
        of samples, to unfold each sample, set skip_begin=1.
        This would, for each i in ``range(tensor.shape[0])``, unfold ``tensor[i, ...]``.

    Parameters
    ----------
    input
        tensor of shape n_samples x n_1 x n_2 x ... x n_i
    mode
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end
    ravel_tensors
        if True, the unfolded tensors are also flattened

    Returns
    -------
    ret
        partially unfolded tensor
    """
    if ravel_tensors:
        new_shape = [-1]
    else:
        new_shape = [input.shape[mode + skip_begin], -1]

    if skip_begin:
        new_shape = [input.shape[i] for i in range(skip_begin)] + new_shape

    if skip_end:
        new_shape += [input.shape[-i] for i in range(1, 1 + skip_end)]

    return ivy.reshape(
        ivy.moveaxis(input, mode + skip_begin, skip_begin), new_shape, out=out
    )


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_fold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: int,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Re-folds a partially unfolded tensor.

    Parameters
    ----------
    input
        a partially unfolded tensor
    mode
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    shape
        the shape of the original full tensor (including skipped dimensions)
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end

    Returns
    -------
    ret
        partially re-folded tensor
    """
    transposed_shape = list(shape)
    mode_dim = transposed_shape.pop(skip_begin + mode)
    transposed_shape.insert(skip_begin, mode_dim)
    return ivy.moveaxis(
        ivy.reshape(input, transposed_shape), skip_begin, skip_begin + mode, out=out
    )


def partial_tensor_to_vec(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Partial vectorization of a tensor while ignoring the specified dimension at the
    beginning and the end.

    Parameters
    ----------
    input
        tensor to partially vectorise
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end

    Returns
    -------
    ret
        partially vectorised tensor with the
        `skip_begin` first and `skip_end` last dimensions untouched
    """
    return partial_unfold(
        input,
        mode=0,
        skip_begin=skip_begin,
        skip_end=skip_end,
        ravel_tensors=True,
        out=out,
    )


def partial_vec_to_tensor(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Refolds a partially vectorised tensor into a full one.

    Parameters
    ----------
    input
        a partially vectorised tensor
    shape
        the shape of the original full tensor (including skipped dimensions)
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end

    Returns
    -------
    ret
        full tensor
    """
    return partial_fold(
        input, mode=0, shape=shape, skip_begin=skip_begin, skip_end=skip_end, out=out
    )


def matricize(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    row_modes: Tuple[int],
    column_modes: Optional[Tuple[int]] = None,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Matricizes the given tensor.

    Parameters
    ----------
    tensor
    row_modes
        modes to use as row of the matrix (in the desired order)
    column_modes
        modes to use as column of the matrix, in the desired order
        if None, the modes not in `row_modes` will be used in ascending order

    ret
    -------
    matrix : tensor of size (ivy.prod(input.shape[i] for i in row_modes), -1)
    """
    ndims = len(input.shape)
    try:
        row_indices = list(row_modes)
    except TypeError:
        row_indices = [row_modes]

    if column_modes is None:
        column_indices = [i for i in range(ndims) if i not in row_indices]
    else:
        try:
            column_indices = list(column_modes)
        except TypeError:
            column_indices = [column_modes]
        if sorted(column_indices + row_indices) != list(range(ndims)):
            msg = (
                "If you provide both column and row modes for the matricization then"
                " column_modes + row_modes must contain all the modes of the tensor."
                f" Yet, got row_modes={row_modes} and column_modes={column_modes}."
            )
            raise ValueError(msg)

    row_size = ivy.prod(input.shape[i] for i in row_indices)
    column_size = ivy.prod(input.shape[i] for i in column_indices)

    return ivy.reshape(
        ivy.transpose(input, row_indices + column_indices),
        (row_size, column_size),
        out=out,
    )
