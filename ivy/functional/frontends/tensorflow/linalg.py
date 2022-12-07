# local
import ivy


from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back

from ivy.functional.frontends.tensorflow import promote_types_of_tensorflow_inputs
import ivy.functional.frontends.tensorflow as tf_frontend


@to_ivy_arrays_and_back
def matrix_rank(a, tol=None, validate_args=False, name=None):
    return ivy.astype(ivy.matrix_rank(a, atol=tol), ivy.int32)


@to_ivy_arrays_and_back
def det(input, name=None):
    return ivy.det(input)


@to_ivy_arrays_and_back
def eigh(tensor, name=None):
    return ivy.eigh(tensor)


@to_ivy_arrays_and_back
def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
def solve(matrix, rhs):
    matrix, rhs = promote_types_of_tensorflow_inputs(matrix, rhs)
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
    chol, rhs = promote_types_of_tensorflow_inputs(chol, rhs)
    y = ivy.solve(chol, rhs)
    return ivy.solve(ivy.matrix_transpose(chol), y)


@to_ivy_arrays_and_back
def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rtol=rcond)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.9.0 and below": ("float32", "float64", "int32")}, "tensorflow"
)
def tensordot(a, b, axes, name=None):
    a, b = promote_types_of_tensorflow_inputs(a, b)
    return ivy.tensordot(a, b, axes=axes)


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
    return ivy.trace(x)


@to_ivy_arrays_and_back
def matrix_transpose(a, name="matrix_transpose", conjugate=False):
    # Conjugate is ignored - Should be added as an argument
    # if complex numbers become supported
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
