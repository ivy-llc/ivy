import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch import promote_types_of_torch_inputs


@to_ivy_arrays_and_back
def atleast_1d(*tensors):
    return ivy.atleast_1d(*tensors)


@to_ivy_arrays_and_back
def flip(input, dims):
    return ivy.flip(input, axis=dims)


@to_ivy_arrays_and_back
def fliplr(input):
    ivy.utils.assertions.check_greater(
        len(input.shape),
        2,
        allow_equal=True,
        message="requires tensor to be at least 2D",
        as_array=False,
    )
    return ivy.fliplr(input)


@to_ivy_arrays_and_back
def flipud(input):
    ivy.utils.assertions.check_greater(
        len(input.shape),
        1,
        allow_equal=True,
        message="requires tensor to be at least 1D",
        as_array=False,
    )
    return ivy.flipud(input)


@to_ivy_arrays_and_back
def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, axis=dims)


@to_ivy_arrays_and_back
def meshgrid(*tensors, indexing=None):
    if indexing is None:
        indexing = "ij"
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]
    return tuple(ivy.meshgrid(*tensors, indexing=indexing))


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.0.1 and below": ("uint8", "bfloat16", "float16"), "1.12.1": ()},
    "torch",
)
def cumsum(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = ivy.int64
    return ivy.cumsum(input, axis=dim, dtype=dtype, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def trace(input):
    if "int" in input.dtype:
        input = input.astype("int64")
    target_type = "int64" if "int" in input.dtype else input.dtype
    return ivy.astype(ivy.trace(input), target_type)


@with_unsupported_dtypes({"2.0.1 and below": ("int8", "uint8", "int16")}, "torch")
@to_ivy_arrays_and_back
def tril_indices(row, col, offset=0, *, dtype=ivy.int64, device="cpu", layout=None):
    sample_matrix = ivy.tril(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


@to_ivy_arrays_and_back
def cummax(input, dim, *, out=None):
    return ivy.cummax(input, axis=dim, out=out)


@to_ivy_arrays_and_back
def cumprod(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = ivy.int64
    return ivy.cumprod(input, axis=dim, dtype=dtype, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def diagonal(input, offset=0, dim1=0, dim2=1):
    return ivy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


@to_ivy_arrays_and_back
def cartesian_prod(*tensors):
    if len(tensors) == 1:
        return tensors

    ret = ivy.meshgrid(*tensors, indexing="ij")
    ret = ivy.stack(ret, axis=-1)
    ret = ivy.reshape(ret, shape=(-1, len(tensors)))

    return ret


@to_ivy_arrays_and_back
def triu_indices(row, col, offset=0, dtype="int64", device="cpu", layout=None):
    # TODO: Handle layout flag when possible.
    sample_matrix = ivy.triu(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


@with_supported_dtypes(
    {"2.5.0 and below": ("float64", "float32", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def triu(input, diagonal=0, *, out=None):
    return ivy.triu(input, k=diagonal, out=out)


@with_supported_dtypes({"2.5.0 and below": ("int8", "int16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tril(input, diagonal=0, *, out=None):
    return ivy.tril(input, k=diagonal, out=out)


@to_ivy_arrays_and_back
def flatten(input, start_dim=0, end_dim=-1):
    return ivy.flatten(input, start_dim=start_dim, end_dim=end_dim)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def renorm(input, p, dim, maxnorm, *, out=None):
    # Torch hardcodes this magic number
    epsilon = 1e-07

    # To iterate through the n-th dimension of `input`, it is easiest to swap
    # the dimension that we wish to iterate through to be first, then iterate
    # through the re-ordered data. This re-ordering is fine for our purposes
    # as we calculate the p-norms and they are all order agnostic. That is,
    # we may re-order the elements of any vector, and as long as none are
    # added, edited, or removed, the p-norm will be the same.
    input_swapped = ivy.swapaxes(input, 0, dim)
    individual_tensors = [input_swapped[i, ...] for i in range(input_swapped.shape[0])]
    ret = []
    for individual_tensor in individual_tensors:
        # These tensors may be multidimensional, but must be treated as a single vector.
        original_shape = individual_tensor.shape
        tensor_flattened = ivy.flatten(individual_tensor)

        # Don't scale up to the maximum norm, only scale down to it.
        norm = ivy.vector_norm(tensor_flattened, axis=0, ord=p)
        multiplier = ivy.minimum(maxnorm / (norm + epsilon), ivy.ones_like(norm))

        # Store the result in its original shape
        ret.append(
            ivy.reshape(ivy.multiply(tensor_flattened, multiplier), original_shape)
        )

    # We must undo our axis swap from the start.
    ret = ivy.asarray(ret, dtype=ret[0].dtype)
    ret = ivy.swapaxes(ret, 0, dim)
    ret = ivy.reshape(ret, input.shape)

    if ivy.exists(out):
        ivy.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "bfloat16",
            "integer",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def logcumsumexp(input, dim, *, out=None):
    if len(input.shape) == 0:
        ret = input
    else:
        # For numerical stability, cast to float64
        # We cast back to the original type at the end.
        original_dtype = input.dtype
        exp_input = ivy.exp(input.astype("float64"))
        summed_exp_input = ivy.cumsum(exp_input, axis=dim)
        ret = ivy.log(summed_exp_input).astype(original_dtype)
    if ivy.exists(out):
        ivy.inplace_update(out, ret)
    return ret


@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "int32",
            "int64",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    return ivy.repeat(input, repeats, axis=dim)


@to_ivy_arrays_and_back
def ravel(input):
    return ivy.reshape(input, (-1,))


@to_ivy_arrays_and_back
def rot90(input, k, dims):
    total_dims = ivy.get_num_dims(input)
    total_rot_dims = len(dims)

    ivy.utils.assertions.check_greater(
        total_dims,
        2,
        allow_equal=True,
        message="expected total dims >= 2, but got total dims = " + str(total_dims),
        as_array=False,
    )

    ivy.utils.assertions.check_equal(
        total_rot_dims,
        2,
        message="expected total rotation dims == 2, but got dims = "
        + str(total_rot_dims),
        as_array=False,
    )

    ivy.utils.assertions.check_equal(
        dims[0],
        dims[1],
        inverse=True,
        message="expected rotation dims to be different, but got dim0 = "
        + str(dims[0])
        + " and dim1 = "
        + str(dims[1]),
        as_array=False,
    )

    ivy.utils.assertions.check_equal(
        ivy.abs(dims[0] - dims[1]),
        total_dims,
        inverse=True,
        message="expected rotation dims to be different, but got dim0 = "
        + str(dims[0])
        + " and dim1 = "
        + str(dims[1]),
    )

    # range of dims
    ivy.utils.assertions.check_less(
        dims[0],
        total_dims,
        message="Rotation dim0 out of range, dim0 = " + str(dims[0]),
        as_array=False,
    )

    ivy.utils.assertions.check_greater(
        dims[0],
        -total_dims,
        allow_equal=True,
        message="Rotation dim0 out of range, dim0 = " + str(dims[0]),
        as_array=False,
    )

    ivy.utils.assertions.check_less(
        dims[1],
        total_dims,
        message="Rotation dim1 out of range, dim1 = " + str(dims[1]),
        as_array=False,
    )

    ivy.utils.assertions.check_greater(
        dims[1],
        -total_dims,
        allow_equal=True,
        message="Rotation dim1 out of range, dim1 = " + str(dims[1]),
        as_array=False,
    )

    k = (4 + (k % 4)) % 4
    new_axes = list(range(total_dims))
    new_axes[min(dims)], new_axes[max(dims)] = max(dims), min(dims)
    if k == 1:
        flipped = ivy.flip(input, axis=dims[1])
        return ivy.permute_dims(flipped, axes=new_axes)
    elif k == 2:
        return ivy.flip(input, axis=dims)
    elif k == 3:
        flipped = ivy.flip(input, axis=dims[0])
        return ivy.permute_dims(flipped, axes=new_axes)
    else:
        return input


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    # if N == 0:
    #     return ivy.array([], dtype=x.dtype)
    # else:
    return ivy.vander(x, N=N, increasing=increasing, out=None)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("int8",)}, "torch")
def lcm(input, other, *, out=None):
    return ivy.lcm(input, other, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    return ivy.einsum(equation, *operands)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def cross(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -1
    input, other = promote_types_of_torch_inputs(input, other)
    return ivy.cross(input, other, axisa=-1, axisb=-1, axisc=-1, axis=dim, out=out)


@to_ivy_arrays_and_back
def gcd(input, other, *, out=None):
    return ivy.gcd(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def tensordot(a, b, dims=2, out=None):
    a, b = promote_types_of_torch_inputs(a, b)
    return ivy.tensordot(a, b, axes=dims, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.0.1 and below": ("int8", "float16", "bfloat16", "bool")}, "torch"
)
def diff(input, n=1, dim=-1, prepend=None, append=None):
    return ivy.diff(input, n=n, axis=dim, prepend=prepend, append=append, out=None)


@to_ivy_arrays_and_back
def broadcast_shapes(*shapes):
    return ivy.broadcast_shapes(*shapes)


@to_ivy_arrays_and_back
def atleast_2d(*tensors):
    return ivy.atleast_2d(*tensors)


@to_ivy_arrays_and_back
def searchsorted(
    sorted_sequence,
    values,
    /,
    *,
    out_int32=False,
    right=False,
    side="left",
    out=None,
    sorter=None,
):
    if right and side == "left":
        raise ivy.exceptions.IvyError(
            "side and right can't be set to opposites, got side of left"
            " while right was True"
        )
    if right:
        side = "right"
    ret = ivy.searchsorted(sorted_sequence, values, side=side, out=out, sorter=sorter)
    if out_int32:
        ret = ivy.astype(ret, "int32")
    return ret


@to_ivy_arrays_and_back
def atleast_3d(*tensors):
    return ivy.atleast_3d(*tensors)


@to_ivy_arrays_and_back
def diag(input, diagonal=0, *, out=None):
    return ivy.diag(input, k=diagonal)


@to_ivy_arrays_and_back
def clone(input):
    return ivy.copy_array(input)


@to_ivy_arrays_and_back
def cov(input, /, *, correction=1, fweights=None, aweights=None):
    return ivy.cov(input, ddof=correction, fweights=fweights, aweights=aweights)
