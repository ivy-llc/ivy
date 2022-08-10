# global
import tensorflow as tf
from typing import Union, Optional, Tuple, Literal, List, NamedTuple
from collections import namedtuple

# local
import ivy
from ivy import inf


# Array API Standard #
# -------------------#


def cholesky(
    x: Union[tf.Tensor, tf.Variable],
    upper: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if not upper:
        ret = tf.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = tf.transpose(tf.linalg.cholesky(tf.transpose(x, perm=axes)), perm=axes)
    return ret


cholesky.unsupported_dtypes = ("float16",)


def cross(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.experimental.numpy.cross(x1, x2, axis=axis)
    return ret


def det(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.linalg.det(x)
    return ret


det.unsupported_dtypes = ("float16",)


def diagonal(
    x: Union[tf.Tensor, tf.Variable],
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.experimental.numpy.diagonal(x, offset, axis1=axis1, axis2=axis2)
    return ret


def eigh(x: Union[tf.Tensor, tf.Variable]) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.linalg.eigh(x)
    return ret


eigh.unsupported_dtypes = ("float16",)


def eigvalsh(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.linalg.eigvalsh(x)
    return ret


eigvalsh.unsupported_dtypes = ("float16",)


def inv(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if tf.math.reduce_any(tf.linalg.det(x) == 0):
        ret = x
    else:
        ret = tf.linalg.inv(x)
    return ret


inv.unsupported_dtypes = ("float16",)


def matmul(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    dtype_from = tf.experimental.numpy.promote_types(
        x1.dtype.as_numpy_dtype, x2.dtype.as_numpy_dtype
    )
    dtype_from = tf.as_dtype(dtype_from)
    if dtype_from.is_unsigned or dtype_from == tf.int8 or dtype_from == tf.int16:
        x1 = tf.cast(x1, tf.int64)
        x2 = tf.cast(x2, tf.int64)
    if x1.dtype != x2.dtype:
        x1 = tf.cast(x1, dtype_from)
        x2 = tf.cast(x2, dtype_from)

    if (
        x1.shape == ()
        or x2.shape == ()
        or (len(x1.shape) == len(x2.shape) == 1 and x1.shape != x2.shape)
        or (len(x1.shape) == len(x2.shape) == 1 and x1.shape != x2.shape)
        or (len(x1.shape) == 1 and len(x2.shape) >= 2 and x1.shape[0] != x2.shape[-2])
        or (len(x2.shape) == 1 and len(x1.shape) >= 2 and x2.shape[0] != x1.shape[-1])
        or (len(x1.shape) >= 2 and len(x2.shape) >= 2 and x1.shape[-1] != x2.shape[-2])
    ):
        raise Exception("Error,shapes not compatible")

    x1_padded = False
    x1_padded_2 = False
    x2_padded = False

    if len(x1.shape) == len(x2.shape) == 1:
        if x1.shape == 0:
            ret = tf.constant(0)
        else:

            ret = tf.reduce_sum(tf.math.multiply(x1, x2))
        ret = tf.cast(ret, dtype=dtype_from)  # return ret

    else:
        if len(x1.shape) == 1:
            if len(x2.shape) == 2:
                x1_padded_2 = True
            elif len(x2.shape) > 2:
                x1_padded = True
            x1 = tf.expand_dims(x1, axis=0)

        elif len(x2.shape) == 1 and len(x1.shape) >= 2:
            x2 = tf.expand_dims(x2, axis=1)
            x2_padded = True

        ret = tf.matmul(x1, x2)

    ret = tf.cast(ret, dtype=dtype_from)
    if x1_padded_2:
        ret = ret[0]
    elif x1_padded:
        ret = tf.squeeze(ret, axis=-2)
    elif x2_padded:
        ret = tf.squeeze(ret, axis=-1)
    return ret


def matrix_norm(
    x: Union[tf.Tensor, tf.Variable],
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    axes = (-2, -1)
    if ord == -float("inf"):
        ret = tf.reduce_min(
            tf.reduce_sum(tf.abs(x), axis=axes[1], keepdims=True), axis=axes
        )
    elif ord == -1:
        ret = tf.reduce_min(
            tf.reduce_sum(tf.abs(x), axis=axes[0], keepdims=True), axis=axes
        )
    elif ord == -2:
        ret = tf.reduce_min(x, axis=(-2, -1), keepdims=keepdims)
    elif ord == "nuc":
        if tf.size(x).numpy() == 0:
            ret = x
        else:
            ret = tf.reduce_sum(tf.linalg.svd(x, compute_uv=False), axis=-1)
    else:
        ret = tf.linalg.norm(x, ord, axes, keepdims)

    if keepdims:
        ret = tf.reshape(ret, x.shape[:-2] + (1, 1))
    else:
        ret = tf.reshape(ret, x.shape[:-2])
    return ret


matrix_norm.unsupported_dtypes = ("float16",)


def matrix_power(
    x: Union[tf.Tensor, tf.Variable],
    n: int,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if n == 0:
        return tf.broadcast_to(tf.eye(x.shape[-2], dtype=x.dtype), x.shape)
    elif n < 0:
        x = tf.linalg.inv(x)
        n = abs(n)

    if n == 1:
        return x
    elif n == 2:
        return x @ x
    elif n == 3:
        return (x @ x) @ x

    z = result = None
    while n > 0:
        z = x if z is None else (z @ z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else (result @ z)
    # replace any -0 with 0
    result = tf.where(tf.equal(result, -0), tf.zeros_like(result), result)
    return result


matrix_power.unsupported_dtypes = ("int8", "float16")


# noinspection PyPep8Naming
def matrix_rank(
    x: Union[tf.Tensor, tf.Variable],
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if rtol is None:
        ret = tf.linalg.matrix_rank(x, rtol)
    else:
        x, rtol = ivy.promote_types_of_inputs(x, rtol)
        ret = tf.linalg.matrix_rank(x, rtol)
    ret = tf.cast(ret, ivy.default_int_dtype(as_native=True))
    return ret


def matrix_transpose(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.experimental.numpy.swapaxes(x, -1, -2)
    return ret


matrix_transpose.unsupported_dtypes = (
    "float16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


# noinspection PyUnusedLocal,PyShadowingBuiltins
def outer(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.outer(x1, x2)


def pinv(
    x: Union[tf.Tensor, tf.Variable],
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if rtol is None:
        ret = tf.linalg.pinv(x)
    else:
        ret = tf.linalg.pinv(x, rtol)
    return ret


pinv.unsupported_dtypes = ("float16",)


def qr(x: Union[tf.Tensor, tf.Variable], mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    if mode == "reduced":
        q, r = tf.linalg.qr(x, full_matrices=False)
        ret = res(q, r)
    elif mode == "complete":
        q, r = tf.linalg.qr(x, full_matrices=True)
        ret = res(q, r)
    else:
        raise Exception(
            "Only 'reduced' and 'complete' qr modes are allowed "
            "for the tensorflow backend."
        )
    return ret


qr.unsupported_dtypes = ("float16",)


def slogdet(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable, Tuple[tf.Tensor, ...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = tf.linalg.slogdet(x)
    ret = results(sign, logabsdet)
    return ret


slogdet.unsupported_dtypes = ("float16",)


def solve(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    expanded_last = False
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = tf.expand_dims(x2, axis=1)
    output_shape = tuple(tf.broadcast_static_shape(x1.shape[:-2], x2.shape[:-2]))

    # in case any of the input arrays are empty
    is_empty_x1 = tf.equal(tf.size(x1), 0)
    is_empty_x2 = tf.equal(tf.size(x2), 0)
    if is_empty_x1 or is_empty_x2:
        for i in range(len(x1.shape) - 2):
            x2 = tf.expand_dims(x2, axis=0)
        output_shape = list(output_shape)
        output_shape.append(x2.shape[-2])
        output_shape.append(x2.shape[-1])
        ret = tf.constant([])
        ret = tf.reshape(ret, output_shape)
    else:
        x1 = tf.broadcast_to(x1, output_shape + x1.shape[-2:])
        x2 = tf.broadcast_to(x2, output_shape + x2.shape[-2:])
        ret = tf.linalg.solve(x1, x2)

    if expanded_last:
        ret = tf.squeeze(ret, axis=-1)
    return ret


solve.unsupported_dtypes = ("float16",)


def svd(
    x: Union[tf.Tensor, tf.Variable],
    full_matrices: bool = True,
) -> Union[tf.Tensor, tf.Variable, Tuple[tf.Tensor, ...]]:
    results = namedtuple("svd", "U S Vh")

    batch_shape = tf.shape(x)[:-2]
    num_batch_dims = len(batch_shape)
    transpose_dims = list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims]
    D, U, V = tf.linalg.svd(x, full_matrices=full_matrices)
    VT = tf.transpose(V, transpose_dims)
    ret = results(U, D, VT)
    return ret


svd.unsupported_dtypes = ("float16",)


def svdvals(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.linalg.svd(x, compute_uv=False)
    return ret


svdvals.unsupported_dtypes = ("float16",)


def tensordot(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    # find type to promote to
    dtype = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)

    # type casting to float32 which is acceptable for tf.tensordot
    x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)

    ret = tf.cast(tf.tensordot(x1, x2, axes), dtype)
    return ret


def trace(
    x: Union[tf.Tensor, tf.Variable],
    offset: int = 0,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.experimental.numpy.trace(
        x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype
    )
    return ret


trace.unsupported_dtypes = ("float16",)


def vecdot(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    dtype = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)
    x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)
    ret = tf.cast(tf.tensordot(x1, x2, (axis, axis)), dtype)
    return ret


vecdot.unsupported_dtypes = ("int8",)


def vector_norm(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if ord == -float("inf"):
        tn_normalized_vector = tf.reduce_min(tf.abs(x), axis, keepdims)
    elif ord < 1:
        tn_normalized_vector = tf.reduce_sum(tf.abs(x) ** ord, axis, keepdims) ** (
            1.0 / ord
        )

    elif ord == 0:
        tn_normalized_vector = tf.reduce_sum(tf.cast(x != 0, x.dtype), axis, keepdims)

    else:
        tn_normalized_vector = tf.linalg.norm(x, ord, axis, keepdims)

    if tn_normalized_vector.shape == tuple():
        ret = tf.expand_dims(tn_normalized_vector, 0)
    else:
        ret = tn_normalized_vector
    return ret


vector_norm.unsupported_dtypes = ("float16",)


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(
    vector: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = tf.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = tf.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = tf.concat((zs, -a3s, a2s), -1)
    row2 = tf.concat((a3s, zs, -a1s), -1)
    row3 = tf.concat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    ret = tf.concat((row1, row2, row3), -2)
    return ret


vector_to_skew_symmetric_matrix.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float64",
)
# vector_to_skew_symmetric_matrix.unsupported_dtypes = ("float16", "float64")
