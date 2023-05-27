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
def matrix_rank(a, tol=None, validate_args=False, name=None):
    return ivy.astype(ivy.matrix_rank(a, atol=tol), ivy.int32)


@to_ivy_arrays_and_back
def det(input, name=None):
    return ivy.det(input)


@to_ivy_arrays_and_back
def eig(tensor, name=None):
    return ivy.eig(tensor)


@to_ivy_arrays_and_back
def eigh(tensor, name=None):
    return ivy.eigh(tensor)


@to_ivy_arrays_and_back
def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.9.0 and below": (
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
@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
def solve(matrix, rhs):
    matrix, rhs = check_tensorflow_casting(matrix, rhs)
    return ivy.solve(matrix, rhs)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.9.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
def logdet(matrix, name=None):
    return ivy.det(matrix).log()


@to_ivy_arrays_and_back
def slogdet(input, name=None):
    return ivy.slogdet(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
def cholesky_solve(chol, rhs, name=None):
    chol, rhs = check_tensorflow_casting(chol, rhs)
    y = ivy.solve(chol, rhs)
    return ivy.solve(ivy.matrix_transpose(chol), y)


def lu_solve(lower_upper, perm, rhs, validate_args=False, name=None):
    """
        Solve a system of linear equations using LU decomposition.

        This function takes a LU decomposed matrix (lower_upper), a permutation matrix (perm), and a right-hand side vector (rhs), and solves the system for the unknown vector.

        Parameters:
        lower_upper (ivy.Array): An array representing the LU decomposition of a matrix A.
        perm (ivy.Array): A permutation matrix from the LU decomposition.
        rhs (ivy.Array): The right-hand side vector in the system of equations.
        validate_args (bool, optional): If set to True, the function will validate the input arguments.
        name (str, optional): The name for the operation.

        Raises:
        ValueError: If the last two dimensions of 'lower_upper' and 'rhs' don't match, or if the last dimension of 'lower_upper' doesn't match the size of 'perm'.

        Returns:
        ivy.Array: The solution to the system of equations Ax = rhs.

        Examples:
        >>> import ivy
        >>> lower_upper = ivy.array([[4,3],[6,3]])
        >>> perm = ivy.array([0,1])
        >>> rhs = ivy.array([7,8])
        >>> x = lu_solve(lower_upper, perm, rhs)
        """
    if validate_args:
        if lower_upper.shape[-2:] != rhs.shape[-2:]:
            raise ValueError("Last two dimensions of 'lower_upper' and 'rhs' must match.")
        if lower_upper.shape[-1] != perm.shape[-1]:
            raise ValueError("'lower_upper' last dimension must match 'perm' size.")

    # extract L and U matrices from lower_upper
    n = lower_upper.shape[-1]
    eye = ivy.eye(n, dtype=lower_upper.dtype)
    L = ivy.tril(lower_upper) - eye + ivy.eye(n, dtype=lower_upper.dtype)
    U = ivy.triu(lower_upper)

    # permute the right hand side (rhs)
    perm_rhs = ivy.take_along_axis(rhs, perm[..., None], axis=-2)

    # solve the lower triangular system
    y = ivy.solve(L, perm_rhs)

    # solve the upper triangular system
    x = ivy.solve(U, y)

    return x

@to_ivy_arrays_and_back
def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rtol=rcond)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.9.0 and below": ("float32", "float64", "int32")}, "tensorflow"
)
def tensordot(a, b, axes, name=None):
    a, b = check_tensorflow_casting(a, b)
    if not ivy.isscalar(axes):
        axes = ivy.to_list(axes)
    return ivy.tensordot(a, b, axes=axes)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.12.0 and below": (
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


@handle_tf_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
def eye(num_rows, num_columns=None, batch_shape=None, dtype=ivy.float32, name=None):
    return ivy.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)


@to_ivy_arrays_and_back
def norm(tensor, ord="euclidean", axis=None, keepdims=None, name=None):
    keepdims = keepdims or False

    # Check if it's a matrix norm
    if (type(axis) in [tuple, list]) and (len(axis) == 2):
        return ivy.matrix_norm(tensor, ord=ord, axis=axis, keepdims=keepdims)
    # Else resort to a vector norm
    return ivy.vector_norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


norm.supported_dtypes = (
    "float32",
    "float64",
)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.9.0 and below": ("float32", "float64")}, "tensorflow")
def normalize(tensor, ord="euclidean", axis=None, name=None):
    tensor = tf_frontend.convert_to_tensor(
        tensor, dtype=ivy.dtype(tensor), dtype_hint="Any"
    )
    _norm = norm(tensor, ord=ord, axis=axis, keepdims=True)
    normalized = tf_frontend.math.divide(tensor, _norm)
    return normalized, _norm


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.9.0 and below": ("float32", "float64")}, "tensorflow")
def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    square_sum = ivy.sum(ivy.square(x), axis=axis, keepdims=True)
    x_inv_norm = ivy.reciprocal(ivy.sqrt(ivy.maximum(square_sum, epsilon)))
    return ivy.multiply(x, x_inv_norm)


@to_ivy_arrays_and_back
def trace(x, name=None):
    return ivy.trace(x, axis1=-2, axis2=-1)


@to_ivy_arrays_and_back
def matrix_transpose(a, name="matrix_transpose", conjugate=False):
    if conjugate:
        return ivy.adjoint(a)
    return ivy.matrix_transpose(a)


@to_ivy_arrays_and_back
def global_norm(t_list, name=None):
    l2_norms = [
        ivy.sqrt((ivy.sum(ivy.square(t)))) ** 2 for t in t_list if t is not None
    ]
    return ivy.sqrt(ivy.sum(ivy.asarray(l2_norms, dtype=ivy.dtype(l2_norms[0]))))


global_norm.supported_dtypes = (
    "float32",
    "float64",
)


@to_ivy_arrays_and_back
def cholesky(input, name=None):
    def symmetrize(input):
        # TODO : Take Hermitian transpose after complex numbers added
        return (input + ivy.swapaxes(input, -1, -2)) / 2

    input = symmetrize(input)

    return ivy.cholesky(input)


@to_ivy_arrays_and_back
def cross(a, b, name=None):
    return ivy.cross(a, b)


@to_ivy_arrays_and_back
def svd(a, /, *, full_matrices=False, compute_uv=True, name=None):
    return ivy.svd(a, compute_uv=compute_uv, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
    return ivy.lu_matrix_inverse(
        ivy.lu_reconstruct(lower_upper, perm), validate_args=validate_args, name=name
    )


@to_ivy_arrays_and_back
def einsum(equation, *inputs, **kwargs):
    return tf_frontend.einsum(equation, *inputs, **kwargs)


@to_ivy_arrays_and_back
def adjoint(matrix, name=None):
    return ivy.adjoint(matrix)


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
