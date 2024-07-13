# local
import ivy
from ivy.functional.frontends.tensorflow import check_tensorflow_casting
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)

import ivy.functional.frontends.tensorflow as tf_frontend


@to_ivy_arrays_and_back
def adjoint(matrix, name=None):
    return ivy.adjoint(matrix)


@to_ivy_arrays_and_back
def band_part(input, num_lower, num_upper, name=None):
    m, n = ivy.meshgrid(
        ivy.arange(input.shape[-2]), ivy.arange(input.shape[-1]), indexing="ij"
    )
    mask = ((num_lower < 0) | ((m - n) <= num_lower)) & (
        (num_upper < 0) | ((n - m) <= num_upper)
    )
    return ivy.where(mask, input, ivy.zeros_like(input))


@to_ivy_arrays_and_back
def cholesky(input, name=None):
    def symmetrize(input):
        # TODO : Take Hermitian transpose after complex numbers added
        return (input + ivy.swapaxes(input, -1, -2)) / 2

    input = symmetrize(input)

    return ivy.cholesky(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, "tensorflow")
def cholesky_solve(chol, rhs, name=None):
    chol, rhs = check_tensorflow_casting(chol, rhs)
    y = ivy.solve(chol, rhs)
    return ivy.solve(ivy.matrix_transpose(chol), y)


@to_ivy_arrays_and_back
def cross(a, b, name=None):
    return ivy.cross(a, b)


@to_ivy_arrays_and_back
def det(input, name=None):
    return ivy.det(input)


@to_ivy_arrays_and_back
def diag(
    diagonal,
    /,
    k=0,
    *,
    num_rows=None,
    num_cols=None,
    padding_value=0,
    align="RIGHT_LEFT",
    name="diag",
):
    # TODO: Implement ivy.matrix_diag in ivy API
    diagonal = ivy.array(diagonal)
    shape = list(diagonal.shape)
    shape[-1] += abs(k)

    output = ivy.full(shape + [shape[-1]], padding_value)
    if k > 0:
        for i in range(shape[-1]):
            try:
                output[..., i, i + k] = diagonal[..., i]
            except IndexError:
                break

    else:
        for i in range(shape[-1]):
            try:
                output[..., i + abs(k), i] = diagonal[..., i]
            except IndexError:
                break

    size = 1
    for dim in output.shape:
        size *= dim
    if (num_cols and num_rows) and (size == (num_cols * num_rows)):
        output = ivy.reshape(output, (num_rows, num_cols))
    return ivy.astype(output, ivy.dtype(diagonal))


@to_ivy_arrays_and_back
def eig(tensor, name=None):
    return ivy.eig(tensor)


@to_ivy_arrays_and_back
def eigh(tensor, name=None):
    return ivy.eigh(tensor)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.15.0 and below": ("float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
def eigvals(tensor, name=None):
    return ivy.eigvals(tensor)


@to_ivy_arrays_and_back
def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


@to_ivy_arrays_and_back
def einsum(equation, *inputs, **kwargs):
    return tf_frontend.einsum(equation, *inputs, **kwargs)


def expm(input, name=None):
    return ivy.matrix_exp(input)


@handle_tf_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, "tensorflow")
def eye(num_rows, num_columns=None, batch_shape=None, dtype=ivy.float32, name=None):
    return ivy.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)


@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
@to_ivy_arrays_and_back
def global_norm(t_list, name=None):
    l2_norms = [ivy.sqrt(ivy.sum(ivy.square(t))) ** 2 for t in t_list if t is not None]
    return ivy.sqrt(ivy.sum(ivy.asarray(l2_norms, dtype=ivy.dtype(l2_norms[0]))))


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def inv(input, adjoint=False, name=None):
    return ivy.inv(input, adjoint=adjoint)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    square_sum = ivy.sum(ivy.square(x), axis=axis, keepdims=True)
    x_inv_norm = ivy.reciprocal(ivy.sqrt(ivy.maximum(square_sum, epsilon)))
    return ivy.multiply(x, x_inv_norm)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
def logdet(matrix, name=None):
    return ivy.det(matrix).log()


@to_ivy_arrays_and_back
def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
    return ivy.lu_matrix_inverse(
        ivy.lu_reconstruct(lower_upper, perm), validate_args=validate_args, name=name
    )


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "float32",
            "float64",
            "int32",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None,
):
    if transpose_a and adjoint_a:
        raise ivy.utils.exceptions.IvyException(
            "Only one of `transpose_a` and `adjoint_a` can be True. "
            "Received `transpose_a`=True, `adjoint_a`=True."
        )
    if transpose_b and adjoint_b:
        raise ivy.utils.exceptions.IvyException(
            "Only one of `transpose_b` and `adjoint_b` can be True. "
            "Received `transpose_b`=True, `adjoint_b`=True."
        )
    return ivy.matmul(
        a,
        b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b,
    )


@to_ivy_arrays_and_back
def matrix_rank(a, tol=None, validate_args=False, name=None):
    # TODO:The tests will fail because output shapes mismatch
    # DO NOT for any reason change anything with the backend function
    # all the fixes must be here as the backend function is
    # working as expected and in compliance with Array API
    return ivy.astype(ivy.matrix_rank(a, atol=tol), ivy.int32)


@to_ivy_arrays_and_back
def matrix_transpose(a, name="matrix_transpose", conjugate=False):
    if conjugate:
        return ivy.adjoint(a)
    return ivy.matrix_transpose(a)


@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
@to_ivy_arrays_and_back
def norm(tensor, ord="euclidean", axis=None, keepdims=None, name=None):
    keepdims = keepdims or False

    # Check if it's a matrix norm
    if (type(axis) in [tuple, list]) and (len(axis) == 2):
        return ivy.matrix_norm(tensor, ord=ord, axis=axis, keepdims=keepdims)
    # Else resort to a vector norm
    return ivy.vector_norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
def normalize(tensor, ord="euclidean", axis=None, name=None):
    tensor = tf_frontend.convert_to_tensor(
        tensor, dtype=ivy.dtype(tensor), dtype_hint="Any"
    )
    _norm = norm(tensor, ord=ord, axis=axis, keepdims=True)
    normalized = tf_frontend.math.divide(tensor, _norm)
    return normalized, _norm


@to_ivy_arrays_and_back
def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rtol=rcond)


@to_ivy_arrays_and_back
def qr(input, /, *, full_matrices=False, name=None):
    return ivy.qr(input)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "half",
            "float32",
            "float64",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def set_diag(input, diagonal, /, *, k=0, align="RIGHT_LEFT", name=None):
    # TODO:
    #  1. Add support for different k values and align options
    #  2. Add support for input tensors with ranks larger than 3

    # Convert input and diagonal to Ivy array format
    input, diagonal = map(ivy.array, (input, diagonal))

    # Check if the input tensor has a rank larger than 3
    if input.ndim > 3:
        raise ivy.utils.exceptions.IvyNotImplementedException(
            "Input tensor must have rank less than or equal to 3.\nInput shape:"
            f" {input.shape}"
        )

    # Check if the first dimension of the input and diagonal match
    if input.shape[0] != diagonal.shape[0]:
        raise ivy.utils.exceptions.IvyValueError(
            "Number of diagonal vectors must match the number of matrices in the"
            f" input.\nInput shape: {input.shape}, Diagonal shape: {diagonal.shape}"
        )

    # Handle the case where input is a 2D matrix
    if input.ndim < 3:
        # Check the diagonal length matches the first dimension of the matrix
        if input.shape[0] != diagonal.shape[0]:
            raise ivy.utils.exceptions.IvyValueError(
                "Length of the diagonal vector must match the first dimension of the"
                f" matrix.\nMatrix shape: {input.shape}, Diagonal shape:"
                f" {diagonal.shape}"
            )

        input[range(input.shape[0]), range(input.shape[0])] = diagonal
    else:
        for matrix, new_diagonal in zip(input, diagonal):
            # Check the diagonal length matches the first dimension of the matrix
            if matrix.shape[0] != new_diagonal.shape[0]:
                raise ivy.utils.exceptions.IvyValueError(
                    "Length of the diagonal vector must match the first dimension of"
                    f" the matrix.\nMatrix shape: {matrix.shape}, Diagonal shape:"
                    f" {new_diagonal.shape}"
                )

            matrix[range(matrix.shape[0]), range(matrix.shape[0])] = new_diagonal

    return input


@to_ivy_arrays_and_back
def slogdet(input, name=None):
    return ivy.slogdet(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, "tensorflow")
def solve(matrix, rhs, /, *, adjoint=False, name=None):
    matrix, rhs = check_tensorflow_casting(matrix, rhs)
    return ivy.solve(matrix, rhs, adjoint=adjoint)


@to_ivy_arrays_and_back
def svd(a, /, *, full_matrices=False, compute_uv=True, name=None):
    return ivy.svd(a, compute_uv=compute_uv, full_matrices=full_matrices)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "half",
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def tensor_diag(diagonal, /, *, name=None):
    diagonal = ivy.array(diagonal)
    rank = ivy.matrix_rank(diagonal)
    if rank > 1:
        raise ValueError("wrong tensor rank, at most 1")
    return ivy.diag(diagonal)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def tensor_diag_part(input, name=None):
    shape = ivy.shape(input, as_array=True)
    rank = len(shape)
    if rank % 2 != 0:
        raise ValueError("Wrong tensor rank, rank must be even.")

    rank = len(shape)
    rank_half = int(rank / 2)
    half_shape = shape[:rank_half]
    prod = 1
    for i in range(rank_half):
        if shape[i] != shape[i + rank_half]:
            raise ValueError(
                f"Invalid shape {shape}: dimensions at {i} and {i+rank_half} do not"
                " match."
            )
        prod *= half_shape[i]

    reshaped = ivy.reshape(input, (prod, prod))
    diagonal = ivy.diagonal(reshaped)
    return ivy.reshape(diagonal, tuple(half_shape))


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.15.0 and below": ("float32", "float64", "int32")}, "tensorflow"
)
def tensordot(a, b, axes, name=None):
    a, b = check_tensorflow_casting(a, b)
    if not ivy.isscalar(axes):
        axes = ivy.to_list(axes)
    return ivy.tensordot(a, b, axes=axes)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
    },
    "tensorflow",
)
def tensorsolve(a, b, axes):
    return ivy.tensorsolve(a, b, axes=axes)


@to_ivy_arrays_and_back
def trace(x, name=None):
    return ivy.trace(x, axis1=-2, axis2=-1)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.13.0 and below": (
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
def tridiagonal_solve(
    diagonals,
    rhs,
    diagonals_format="compact",
    transpose_rhs=False,
    conjugate_rhs=False,
    name=None,
    partial_pivoting=True,
    perturb_singular=False,
):
    if transpose_rhs is True:
        rhs_copy = ivy.matrix_transpose(rhs)
    if conjugate_rhs is True:
        rhs_copy = ivy.conj(rhs)
    if not transpose_rhs and not conjugate_rhs:
        rhs_copy = ivy.array(rhs)

    if diagonals_format == "matrix":
        return ivy.solve(diagonals, rhs_copy)
    elif diagonals_format in ["sequence", "compact"]:
        diagonals = ivy.array(diagonals)
        dim = diagonals[0].shape[0]
        diagonals[[0, -1], [-1, 0]] = 0
        dummy_idx = [0, 0]
        indices = ivy.array(
            [
                [(i, i + 1) for i in range(dim - 1)] + [dummy_idx],
                [(i, i) for i in range(dim)],
                [dummy_idx] + [(i + 1, i) for i in range(dim - 1)],
            ]
        )
        constructed_matrix = ivy.scatter_nd(
            indices, diagonals, shape=ivy.array([dim, dim])
        )
        return ivy.solve(constructed_matrix, rhs_copy)
    else:
        raise ValueError("Unexpected diagonals_format")
