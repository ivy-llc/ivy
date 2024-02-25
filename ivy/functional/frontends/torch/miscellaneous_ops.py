import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch import promote_types_of_torch_inputs
import ivy.functional.frontends.torch as torch_frontend


erfinv = torch_frontend.special.erfinv


@to_ivy_arrays_and_back
def atleast_1d(*tensors):
    return ivy.atleast_1d(*tensors)


@to_ivy_arrays_and_back
def atleast_2d(*tensors):
    return ivy.atleast_2d(*tensors)


@to_ivy_arrays_and_back
def atleast_3d(*tensors):
    return ivy.atleast_3d(*tensors)


# TODO: Add Ivy function for block_diag but only scipy.linalg and \
# and torch supports block_diag currently
@to_ivy_arrays_and_back
def block_diag(*tensors):
    shapes_list = [ivy.shape(t) for t in tensors]
    # TODO: Add ivy function to return promoted dtype for multiple tensors at once
    promoted_dtype = ivy.as_ivy_dtype(tensors[0].dtype)
    for idx in range(1, len(tensors)):
        promoted_dtype = torch_frontend.promote_types_torch(
            tensors[idx - 1].dtype, tensors[idx].dtype
        )

    inp_tensors = [ivy.asarray(t, dtype=promoted_dtype) for t in tensors]
    tensors_2d = []
    result_dim_0, result_dim_1 = 0, 0
    for idx, t_shape in enumerate(shapes_list):
        dim_0, dim_1 = 1, 1
        if len(t_shape) > 2:
            raise ivy.exceptions.IvyError(
                "Input tensors must have 2 or fewer dimensions."
                f"Input {idx} has {len(t_shape)} dimensions"
            )
        elif len(t_shape) == 2:
            dim_0, dim_1 = t_shape
            tensors_2d.append(inp_tensors[idx])
        elif len(t_shape) == 1:
            dim_1 = t_shape[0]
            tensors_2d.append(ivy.reshape(inp_tensors[idx], shape=(dim_0, dim_1)))
        else:
            tensors_2d.append(ivy.reshape(inp_tensors[idx], shape=(dim_0, dim_1)))

        result_dim_0 += dim_0
        result_dim_1 += dim_1
        shapes_list[idx] = (dim_0, dim_1)

    ret = ivy.zeros((result_dim_0, result_dim_1), dtype=promoted_dtype)
    ret_dim_0 = 0
    ret_dim_1 = 0
    for idx, t_shape in enumerate(shapes_list):
        dim_0, dim_1 = t_shape
        ret[
            ret_dim_0 : ret_dim_0 + dim_0, ret_dim_1 : ret_dim_1 + dim_1
        ] = ivy.copy_array(tensors_2d[idx])
        ret_dim_0 += dim_0
        ret_dim_1 += dim_1

    return ret


@to_ivy_arrays_and_back
def broadcast_shapes(*shapes):
    return ivy.broadcast_shapes(*shapes)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
@to_ivy_arrays_and_back
def broadcast_to(tensor, shape):
    return ivy.broadcast_to(tensor, shape)


@to_ivy_arrays_and_back
def cartesian_prod(*tensors):
    if len(tensors) == 1:
        return tensors

    ret = ivy.meshgrid(*tensors, indexing="ij")
    ret = ivy.stack(ret, axis=-1)
    ret = ivy.reshape(ret, shape=(-1, len(tensors)))

    return ret


@with_unsupported_dtypes({"2.2 and below": "float16"}, "torch")
@to_ivy_arrays_and_back
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    if len(x1.shape) == 2 and len(x2.shape) == 2:
        x1_first_dim, x2_first_dim = x1.shape[0], x2.shape[0]
        if (
            compute_mode == "use_mm_for_euclid_dist_if_necessary"
            and (x1_first_dim > 25 or x2_first_dim > 25)
            or compute_mode == "use_mm_for_euclid_dist"
        ):
            return ivy.vector_norm(x1[:, None, :] - x2[None, :, :], axis=-1, ord=p)
        else:
            distances = ivy.zeros((x1_first_dim, x2_first_dim), dtype=x1.dtype)
            for i in range(x1_first_dim):
                for j in range(x2_first_dim):
                    distances[i, j] = ivy.vector_norm(x1[i, :] - x2[j, :], ord=p)
            return distances
    if p == 2:
        B, P, M = x1.shape
        _, R, _ = x2.shape
        if (
            compute_mode == "use_mm_for_euclid_dist_if_necessary"
            and (P > 25 or R > 25)
            or compute_mode == "use_mm_for_euclid_dist"
        ):
            return ivy.vector_norm(
                x1[:, :, None, :] - x2[:, None, :, :], axis=-1, ord=p
            )
        else:
            distances = ivy.zeros((B, P, R), dtype=x1.dtype)
            for b in range(B):
                for i in range(P):
                    for j in range(R):
                        distances[b, i, j] = ivy.vector_norm(
                            x1[b, i, :] - x2[b, j, :], ord=p
                        )
            return distances
    else:
        return ivy.vector_norm(x1[:, :, None, :] - x2[:, None, :, :], axis=-1, ord=p)


@to_ivy_arrays_and_back
def clone(input, *, memory_format=None):
    return ivy.copy_array(input)


@with_unsupported_dtypes({"2.2 and below": ("float16", "bool")}, "torch")
@to_ivy_arrays_and_back
def corrcoef(input):
    if len(ivy.shape(input)) > 2:
        raise ivy.exceptions.IvyError(
            "corrcoef(): expected input to have two or fewer dimensions but got an"
            f" input with {ivy.shape(input)} dimensions"
        )
    return ivy.corrcoef(input, y=None, rowvar=True)


@to_ivy_arrays_and_back
def cov(input, /, *, correction=1, fweights=None, aweights=None):
    return ivy.cov(input, ddof=correction, fweights=fweights, aweights=aweights)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def cross(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -1
    input, other = promote_types_of_torch_inputs(input, other)
    return ivy.cross(input, other, axisa=-1, axisb=-1, axisc=-1, axis=dim, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
        )
    },
    "torch",
)
def cummax(input, dim, *, out=None):
    input_dtype = input.dtype
    result_values, result_indices = ivy.cummax(input, axis=dim, out=out)
    result_values = result_values.astype(input_dtype)
    return result_values, result_indices


@to_ivy_arrays_and_back
def cumprod(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = ivy.int64
    return ivy.cumprod(input, axis=dim, dtype=dtype, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.2 and below": ("uint8", "bfloat16", "float16"), "1.12.1": ()},
    "torch",
)
def cumsum(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = ivy.int64
    return ivy.cumsum(input, axis=dim, dtype=dtype, out=out)


@to_ivy_arrays_and_back
def diag(input, diagonal=0, *, out=None):
    return ivy.diag(input, k=diagonal)


@to_ivy_arrays_and_back
def diag_embed(
    input,
    offset=0,
    dim1=-2,
    dim2=-1,
):
    def _handle_dim(rank, idx):
        if idx >= 0 and idx < rank:
            return idx
        if idx < 0:
            idx = idx + rank
        if idx < 0 or idx >= rank:
            raise IndexError
        return idx

    input_type = ivy.dtype(input)
    rank = input.ndim + 1
    dim1 = _handle_dim(rank, dim1)
    dim2 = _handle_dim(rank, dim2)
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset
    last_dim = list(input.shape)[-1]
    if offset != 0:
        # add padding to match the new size
        t_shape = list(input.shape)
        t_shape[-1] = abs(offset)
        z = ivy.zeros(t_shape, dtype=input.dtype, device=input.device)
        pair = (z, input) if offset > 0 else (input, z)
        input = ivy.concat(pair, axis=-1)
        last_dim += abs(offset)
    input = input.expand_dims(axis=dim1).moveaxis(-1, dim2)
    # generate ranges shifting indices based on offset
    a_range = ivy.arange(last_dim, device=input.device, dtype=ivy.int64)
    b_range = ivy.arange(
        offset, last_dim + offset, device=input.device, dtype=ivy.int64
    )
    # broadcast
    cond = a_range == b_range.expand_dims(axis=-1)
    cond_shape = [last_dim if i in (dim1, dim2) else 1 for i in range(len(input.shape))]
    cond = cond.reshape(cond_shape)
    if input.dtype == ivy.bool:
        ret = cond.logical_and(input)
    else:
        ret = ivy.where(cond, input, 0)
    return ret.astype(input_type)


@with_supported_dtypes(
    {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
@to_ivy_arrays_and_back
def diagflat(x, offset=0, name=None):
    arr = ivy.diagflat(x, offset=offset)
    return arr


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def diagonal(input, offset=0, dim1=0, dim2=1):
    return ivy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.2 and below": ("int8", "float16", "bfloat16", "bool")}, "torch"
)
def diff(input, n=1, dim=-1, prepend=None, append=None):
    return ivy.diff(input, n=n, axis=dim, prepend=prepend, append=append, out=None)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    return ivy.einsum(equation, *operands)


@to_ivy_arrays_and_back
def finfo(dtype):
    return ivy.finfo(dtype)


@to_ivy_arrays_and_back
def flatten(input, start_dim=0, end_dim=-1):
    return ivy.flatten(input, start_dim=start_dim, end_dim=end_dim)


@to_ivy_arrays_and_back
def flip(input, dims):
    return ivy.flip(input, axis=dims, copy=True)


@to_ivy_arrays_and_back
def fliplr(input):
    ivy.utils.assertions.check_greater(
        len(input.shape),
        2,
        allow_equal=True,
        message="requires tensor to be at least 2D",
        as_array=False,
    )
    return ivy.fliplr(input, copy=True)


@to_ivy_arrays_and_back
def flipud(input):
    ivy.utils.assertions.check_greater(
        len(input.shape),
        1,
        allow_equal=True,
        message="requires tensor to be at least 1D",
        as_array=False,
    )
    return ivy.flipud(input, copy=True)


@to_ivy_arrays_and_back
def gcd(input, other, *, out=None):
    return ivy.gcd(input, other, out=out)


@to_ivy_arrays_and_back
def kron(input, other, *, out=None):
    return ivy.kron(input, other, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("int8",)}, "torch")
def lcm(input, other, *, out=None):
    return ivy.lcm(input, other, out=out)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
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


@to_ivy_arrays_and_back
def lu_solve(b, LU_data, LU_pivots, *, out=None):
    return torch_frontend.linalg.lu_solve(LU_data, LU_pivots, b, out=out)


@to_ivy_arrays_and_back
def meshgrid(*tensors, indexing=None):
    if indexing is None:
        indexing = "ij"
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]
    return tuple(ivy.meshgrid(*tensors, indexing=indexing))


@to_ivy_arrays_and_back
def ravel(input):
    return ivy.reshape(input, (-1,))


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
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


@with_supported_dtypes(
    {
        "2.2 and below": (
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
def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, axis=dims)


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
        return ivy.permute_dims(flipped, axes=new_axes, copy=True)
    elif k == 2:
        return ivy.flip(input, axis=dims, copy=True)
    elif k == 3:
        flipped = ivy.flip(input, axis=dims[0])
        return ivy.permute_dims(flipped, axes=new_axes, copy=True)
    else:
        return input


@to_ivy_arrays_and_back
def searchsorted(
    sorted_sequence,
    values,
    /,
    *,
    out_int32=False,
    right=False,
    side=None,
    out=None,
    sorter=None,
):
    if side == "left":
        if right:
            raise ivy.exceptions.IvyError(
                "side and right can't be set to opposites, got side of left"
                " while right was True"
            )
    elif side is None:
        side = "right" if right else "left"
    ret = ivy.searchsorted(sorted_sequence, values, side=side, out=out, sorter=sorter)
    if out_int32:
        ret = ivy.astype(ret, "int32")
    return ret


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def tensordot(a, b, dims=2, out=None):
    a, b = promote_types_of_torch_inputs(a, b)
    return ivy.tensordot(a, b, axes=dims, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def trace(input):
    if "int" in input.dtype:
        input = input.astype("int64")
    target_type = "int64" if "int" in input.dtype else input.dtype
    return ivy.astype(ivy.trace(input), target_type)


@with_supported_dtypes({"2.5.0 and below": ("int8", "int16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tril(input, diagonal=0, *, out=None):
    return ivy.tril(input, k=diagonal, out=out)


@with_unsupported_dtypes({"2.2 and below": ("int8", "uint8", "int16")}, "torch")
@to_ivy_arrays_and_back
def tril_indices(row, col, offset=0, *, dtype=ivy.int64, device="cpu", layout=None):
    sample_matrix = ivy.tril(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


@with_supported_dtypes(
    {"2.5.0 and below": ("float64", "float32", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def triu(input, diagonal=0, *, out=None):
    return ivy.triu(input, k=diagonal, out=out)


@to_ivy_arrays_and_back
def triu_indices(row, col, offset=0, dtype="int64", device="cpu", layout=None):
    # TODO: Handle layout flag when possible.
    sample_matrix = ivy.triu(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


@to_ivy_arrays_and_back
def unflatten(input, dim, sizes):
    return ivy.unflatten(input, dim=dim, shape=sizes, out=None)


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    # if N == 0:
    #     return ivy.array([], dtype=x.dtype)
    # else:
    return ivy.vander(x, N=N, increasing=increasing, out=None)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
def view_as_complex(input):
    if ivy.shape(input)[-1] != 2:
        raise ivy.exceptions.IvyError("The last dimension must have a size of 2")

    real, imaginary = ivy.split(
        ivy.stop_gradient(input, preserve_type=False),
        num_or_size_splits=2,
        axis=ivy.get_num_dims(input) - 1,
    )
    dtype = ivy.complex64 if input.dtype == ivy.float32 else ivy.complex128
    real = ivy.squeeze(real, axis=ivy.get_num_dims(real) - 1).astype(dtype)
    imag = ivy.squeeze(imaginary, axis=ivy.get_num_dims(imaginary) - 1).astype(dtype)
    complex_ = real + imag * 1j
    return ivy.array(complex_, dtype=dtype)


@with_supported_dtypes(
    {"2.2 and below": ("complex64", "complex128")},
    "torch",
)
@to_ivy_arrays_and_back
def view_as_real(input):
    if not ivy.is_complex_dtype(input):
        raise ivy.exceptions.IvyError(
            "view_as_real is only supported for complex tensors"
        )
    re_part = ivy.real(input)
    im_part = ivy.imag(input)
    return ivy.stack((re_part, im_part), axis=-1)
