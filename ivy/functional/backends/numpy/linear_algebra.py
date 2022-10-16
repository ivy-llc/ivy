# global

from collections import namedtuple

from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence


import numpy as np

# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output
from . import backend_version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def cholesky(
    x: np.ndarray, /, *, upper: Optional[bool] = False, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not upper:
        ret = np.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = np.transpose(np.linalg.cholesky(np.transpose(x, axes=axes)), axes=axes)
    return ret


def cross(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@_handle_0_dim_output
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def det(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.det(x)


def diag(
    x: np.ndarray,
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[np.ndarray] = None,
):
    # const Tensor& diagonal = context->input(0);
    diagonal = x

    # MatrixDiag and MatrixDiagV2 both use this OpKernel. MatrixDiag only has
    # one input, so we have to check the number of inputs before reading
    # additional parameters in MatrixDiagV2.
    lower_diag_index = 0
    upper_diag_index = 0
    if num_rows is None:
        num_rows = -1
    if num_cols is None:
        num_cols = -1
    # T padding_value(0);

    # // MatrixDiagOpV2-specific.
    # if (context->num_inputs() > kNumV1Inputs) {
    #  auto& diag_index = context->input(1);
    #  OP_REQUIRES(context,
    #              TensorShapeUtils::IsScalar(diag_index.shape()) ||
    #                  TensorShapeUtils::IsVector(diag_index.shape()),
    #              errors::InvalidArgument(
    #                  "diag_index must be a scalar or vector, received shape: ",
    #                  diag_index.shape().DebugString()));
    # OP_REQUIRES(context, diag_index.NumElements() > 0,
    #             errors::InvalidArgument(
    #                  "Expected diag_index to have at least 1 element"));
    #  lower_diag_index = diag_index.flat<int32>()(0);
    # lower_diag_index = 0
    lower_diag_index = offset
    upper_diag_index = lower_diag_index

    # diag_index_size = diagonal.shape[0]
    # if diag_index_size > 1:
    #    upper_diag_index = diagonal[1]

    #  upper_diag_index = lower_diag_index;
    #  if (TensorShapeUtils::IsVector(diag_index.shape())) {
    #    auto diag_index_size = diag_index.dim_size(0);
    #    OP_REQUIRES(
    #        context, 0 < diag_index_size && diag_index_size <= 2,
    #        errors::InvalidArgument(
    #            "diag_index must have only one or two elements, received ",
    #            diag_index_size, " elements."));
    #    if (diag_index_size > 1) {
    #      upper_diag_index = diag_index.flat<int32>()(1);
    #    }
    #  }

    # auto& num_rows_tensor = context->input(2);
    # OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_rows_tensor.shape()),
    #             errors::InvalidArgument("num_rows must be a scalar"));
    # num_rows = num_rows_tensor.flat<int32>()(0);

    # auto& num_cols_tensor = context->input(3);
    # OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_cols_tensor.shape()),
    #            errors::InvalidArgument("num_cols must be a scalar"));
    # num_cols = num_cols_tensor.flat<int32>()(0);

    # auto& padding_value_tensor = context->input(4);
    # OP_REQUIRES(context,
    #            TensorShapeUtils::IsScalar(padding_value_tensor.shape()),
    #            errors::InvalidArgument("padding_value must be a scalar"));
    # padding_value = padding_value_tensor.flat<T>()(0);
    # }

    # // Size validations.
    # const TensorShape& diagonal_shape = diagonal.shape();
    diagonal_shape = diagonal.shape
    # const int diag_rank = diagonal_shape.dims();
    diag_rank = len(diagonal_shape)
    # const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    num_diags = upper_diag_index - lower_diag_index + 1
    # OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diagonal_shape),
    #            errors::InvalidArgument(
    #                "diagonal must be at least 1-dim, received shape: ",
    #                diagonal.shape().DebugString()));
    # OP_REQUIRES(
    #    context, lower_diag_index <= upper_diag_index,
    #    errors::InvalidArgument(
    #        "lower_diag_index must not be larger than upper_diag_index: ",
    #        lower_diag_index, " > ", upper_diag_index));
    # OP_REQUIRES(context,
    #            lower_diag_index == upper_diag_index ||
    #                diagonal_shape.dim_size(diag_rank - 2) == num_diags,
    #            errors::InvalidArgument(
    #                "The number of diagonals provided in the input does not "
    #                "match the lower_diag_index and upper_diag_index range."));

    # const Eigen::Index max_diag_len = diagonal_shape.dim_size(diag_rank - 1);
    max_diag_len = diagonal_shape[diag_rank - 1]
    # const int32_t min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
    min_num_rows = max_diag_len - min(upper_diag_index, 0)
    # const int32_t min_num_cols = max_diag_len + std::max(lower_diag_index, 0);
    min_num_cols = max_diag_len + max(lower_diag_index, 0)

    # OP_REQUIRES(context, num_rows == -1 || num_rows >= min_num_rows,
    #            errors::InvalidArgument("The number of rows is too small."));
    # OP_REQUIRES(context, num_cols == -1 || num_cols >= min_num_cols,
    #            errors::InvalidArgument("The number of columns is too small."));

    # // If both num_rows and num_cols are unknown, assume that output is square.
    # // Otherwise, use smallest possible values.
    # if (num_rows == -1 && num_cols == -1) {
    if num_rows == -1 and num_cols == -1:
        # num_rows = std::max(min_num_rows, min_num_cols);
        num_rows = max(min_num_rows, min_num_cols)
        num_cols = num_rows
    elif num_rows == -1:
        num_rows = min_num_rows
    elif num_cols == -1:
        num_cols = min_num_cols

    # OP_REQUIRES(context, num_rows == min_num_rows || num_cols == min_num_cols,
    #            errors::InvalidArgument(
    #                "The number of rows or columns is not consistent with "
    #                "the specified d_lower, d_upper, and diagonal."));

    # TensorShape output_shape = diagonal_shape;
    output_shape = list(diagonal_shape)
    if num_diags == 1:  # Output has rank `rank+1`.
        # output_shape.set_dim(diag_rank - 1, num_rows);
        output_shape[diag_rank - 1] = num_rows
        # output_shape.AddDim(num_cols);
        output_shape.append(num_cols)
    else:  # Output has rank `rank`.
        # output_shape.set_dim(diag_rank - 2, num_rows);
        output_shape[diag_rank - 2] = num_rows
        # output_shape.set_dim(diag_rank - 1, num_cols);
        output_shape[diag_rank - 1] = num_cols
    # }

    # Tensor* output = nullptr;
    # OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    # auto output_reshaped = output->flat_inner_dims<T, 3>();
    # output_array = np.full((num_rows, num_cols), padding_value)
    output_array = np.full(output_shape, padding_value)
    diag_len = min(num_rows, num_cols) - abs(offset)

    if len(diagonal) < diag_len:
        diagonal = np.array(
            [padding_value] * max((diag_len - len(diagonal), 0)) + list(diagonal)
        )

    if len(diagonal) > diag_len:
        diagonal = diagonal[:diag_len][:diag_len]

    diagonal_to_add = np.diag(
        diagonal - np.full_like(diagonal, padding_value), k=offset
    )
    try:
        # output_array += np.pad(diagonal_to_add,
        # [(0, max(len(diagonal) - diag_len)), (0, max(len(diagonal) - diag_len))],
        # mode="constant")
        output_array += np.pad(
            diagonal_to_add,
            [
                (0, max([output_array.shape[0] - diagonal_to_add.shape[0], 0])),
                (0, max([output_array.shape[1] - diagonal_to_add.shape[1], 0])),
            ],
            mode="constant",
        )
    except Exception as e:
        print("EH")
        print(e)
    return output_array
    # return np.diag()
    # auto diag_reshaped = diagonal.flat<T>();
    # functor::MatrixDiag<Device, T>::Compute(
    #    context, context->eigen_device<Device>(), diag_reshaped,
    #    output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
    #    padding_value, left_align_superdiagonal_, left_align_subdiagonal_);


# }
#    if num_rows is None and num_cols is None:
#        num_rows = len(x) + abs(offset)
#        num_cols = len(x) + abs(offset)
#    if num_rows is None:
#        num_rows = num_cols - offset
#    if num_cols is None:
#        num_cols = num_rows + offset - len(x)#???
#    #if num_rows is None:
#    #    num_rows = len(x) + abs(offset)
#    #if num_cols is None:
#   #    num_cols = len(x) + abs(offset)
#   #ret = np.ones((num_rows, num_cols))
#   #ret *= padding_value
#   ret = np.full((num_rows, num_cols), padding_value, dtype=x.dtype)
#    # Avoiding a later crash where adding a scalar to a matrix is undefined
# padding_value = np.full_like(x, padding_value)###
#
#    diag_length = min(num_rows, num_cols) - abs(offset)

# if diag_length <= len(x):
#    diag_array = x[:diag_length]

# else:
#    diag_array = np.array(
#        ([padding_value] * (diag_length - len(x))) +
#        list(x))

# array_to_add = np.diag(x - np.full_like(x, padding_value), k=offset)
# array_to_add = np.diag(diag_array, k=offset)
# array_to_add = np.diag(diag_array - np.full_like(diag_array, padding_value), k=offset)
# print(f"ARRAY TO ADD IS {array_to_add}")

# array_to_add = np.resize(array_to_add, ret.shape)

# On the diagonal there will be
# 1 * padding_value + x_i - padding_value == x_i
# try:
#    ret += array_to_add
# except Exception as e:
# print("Hehe")
# ret += np.broadcast_to(array_to_add, ret.shape)
# ret += np.diag(x - padding_value, k=offset)

# return ret


def diagonal(
    x: np.ndarray,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def eigh(
    x: np.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", np.ndarray), ("eigenvectors", np.ndarray)]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def eigvalsh(
    x: np.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.eigvalsh(x)


@_handle_0_dim_output
def inner(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.inner(x1, x2)


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def inv(
    x: np.ndarray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if np.any(np.linalg.det(x.astype("float64")) == 0):
        return x
    else:
        if adjoint is False:
            ret = np.linalg.inv(x)
            return ret
        else:
            x = np.transpose(x)
            ret = np.linalg.inv(x)
            return ret


def matmul(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if transpose_a is True:
        x1 = np.transpose(x1)
    if transpose_b is True:
        x2 = np.transpose(x2)
    ret = np.matmul(x1, x2, out=out)
    if len(x1.shape) == len(x2.shape) == 1:
        ret = np.array(ret)
    return ret


matmul.support_native_out = True


@_handle_0_dim_output
@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def matrix_norm(
    x: np.ndarray,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not isinstance(axis, tuple):
        axis = tuple(axis)
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def matrix_power(
    x: np.ndarray, n: int, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.matrix_power(x, n)


@with_unsupported_dtypes(
    {
        "1.23.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
@_handle_0_dim_output
def matrix_rank(
    x: np.ndarray,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(x.shape) < 2:
        return np.asarray(0).astype(x.dtype)
    if type(atol) and type(rtol) == tuple:
        if atol.all() and rtol.all() is None:
            ret = np.asarray(np.linalg.matrix_rank(x, tol=atol)).astype(x.dtype)
        elif atol.all() and rtol.all():
            tol = np.maximum(atol, rtol)
            ret = np.asarray(np.linalg.matrix_rank(x, tol=tol)).astype(x.dtype)
        else:
            ret = np.asarray(np.linalg.matrix_rank(x, tol=rtol)).astype(x.dtype)
    else:
        ret = np.asarray(np.linalg.matrix_rank(x, tol=rtol)).astype(x.dtype)
    return ret


def matrix_transpose(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


def outer(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.outer(x1, x2, out=out)


outer.support_native_out = True


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def pinv(
    x: np.ndarray,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rtol is None:
        return np.linalg.pinv(x)
    else:
        return np.linalg.pinv(x, rtol)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def qr(x: np.ndarray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = np.linalg.qr(x, mode=mode)
    return res(q, r)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def slogdet(
    x: np.ndarray,
    /,
) -> Tuple[np.ndarray, np.ndarray]:
    results = NamedTuple("slogdet", [("sign", np.ndarray), ("logabsdet", np.ndarray)])
    sign, logabsdet = np.linalg.slogdet(x)
    sign = np.asarray(sign) if not isinstance(sign, np.ndarray) else sign
    logabsdet = (
        np.asarray(logabsdet) if not isinstance(logabsdet, np.ndarray) else logabsdet
    )

    return results(sign, logabsdet)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def solve(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = np.expand_dims(x2, axis=1)
    for i in range(len(x1.shape) - 2):
        x2 = np.expand_dims(x2, axis=0)
    ret = np.linalg.solve(x1, x2)
    if expanded_last:
        ret = np.squeeze(ret, axis=-1)
    return ret


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def svd(
    x: np.ndarray, /, *, compute_uv: bool = True, full_matrices: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if compute_uv:
        results = namedtuple("svd", "U S Vh")
        U, D, VT = np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(U, D, VT)
    else:
        results = namedtuple("svd", "S")
        D = np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(D)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def svdvals(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.svd(x, compute_uv=False)


def tensordot(
    x1: np.ndarray,
    x2: np.ndarray,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.tensordot(x1, x2, axes=axes)


@_handle_0_dim_output
def trace(
    x: np.ndarray,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.trace(x, offset=offset, axis1=axis1, axis2=axis2, out=out)


trace.unsupported_dtypes = ("float16", "bfloat16")
trace.support_native_out = True


def vecdot(
    x1: np.ndarray, x2: np.ndarray, axis: int = -1, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.tensordot(x1, x2, axes=(axis, axis))


def vector_norm(
    x: np.ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        np_normalized_vector = np.linalg.norm(x.flatten(), ord, axis, keepdims)

    else:
        np_normalized_vector = np.linalg.norm(x, ord, axis, keepdims)

    if np_normalized_vector.shape == tuple():
        ret = np.expand_dims(np_normalized_vector, 0)
    else:
        ret = np_normalized_vector
    return ret


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(
    vector: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = np.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = np.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = np.concatenate((zs, -a3s, a2s), -1)
    row2 = np.concatenate((a3s, zs, -a1s), -1)
    row3 = np.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return np.concatenate((row1, row2, row3), -2, out=out)


vector_to_skew_symmetric_matrix.support_native_out = True


def vander(
    x: np.ndarray,
    /,
    *,
    N: Optional[int] = None,
    increasing: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vander(x, N=N, increasing=increasing).astype(x.dtype)


vander.support_native_out = False
