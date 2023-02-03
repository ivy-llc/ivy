# global

from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence
from collections import namedtuple

import tensorflow as tf

# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def cholesky(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    upper: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not upper:
        ret = tf.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = tf.transpose(tf.linalg.cholesky(tf.transpose(x, perm=axes)), perm=axes)
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("complex", "float16",)}, backend_version)
def cross(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:

    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.cross(
        x1, x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis
    )


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def det(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.linalg.det(x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def diagonal(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.diagonal(x, offset, axis1=axis1, axis2=axis2)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def eigh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    UPLO: Optional[str] = "L",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[Union[tf.Tensor, tf.Variable]]:

    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    result_tuple = NamedTuple(
        "eigh",
        [
            ("eigenvalues", Union[tf.Tensor, tf.Variable]),
            ("eigenvectors", Union[tf.Tensor, tf.Variable]),
        ],
    )

    if UPLO == "L":
        eigenvalues, eigenvectors = tf.linalg.eigh(x)

    elif UPLO == "U":
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        eigenvalues, eigenvectors = tf.linalg.eigh(tf.transpose(x, perm=axes))
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def eigvalsh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    UPLO: Optional[str] = "L",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    if UPLO == "L":
        return tf.linalg.eigh(x)[0]
    elif UPLO == "U":
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = tf.linalg.eigh(tf.transpose(x, perm=axes))[0]
        return ret


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "int8",
            "uint8",
            "int16",
            "uint16",
            "uint32",
            "uint64",
            "complex"
        )
    },
    backend_version,
)
# noinspection PyUnusedLocal,PyShadowingBuiltins
def inner(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.inner(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def inv(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if tf.math.reduce_any(tf.linalg.det(tf.cast(x, dtype="float64")) == 0):
        return x
    else:
        if adjoint is False:
            ret = tf.linalg.inv(x)
            return ret
        else:
            x = tf.linalg.adjoint(x)
            ret = tf.linalg.inv(x)
            return ret


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def matmul(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:

    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    dtype_from = tf.as_dtype(x1.dtype)

    if transpose_a is True:
        x1 = tf.transpose(x1)
    if transpose_b is True:
        x2 = tf.transpose(x2)

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
        raise ivy.exceptions.IvyException("Error,shapes not compatible")

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

    ret = ivy.astype(ret, dtype_from, copy=False).to_native()
    if x1_padded_2:
        ret = ret[0]
    elif x1_padded:
        ret = tf.squeeze(ret, axis=-2)
    elif x2_padded:
        ret = tf.squeeze(ret, axis=-1)
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def matrix_norm(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ord == -float("inf"):
        reduce_min = tf.reduce_min(
            tf.reduce_sum(tf.abs(x), axis=axis[1], keepdims=True), axis=axis
        )
        ret = reduce_min
    elif ord == -1:
        ret = tf.reduce_min(
            tf.reduce_sum(tf.abs(x), axis=axis[0], keepdims=True), axis=axis
        )
    elif ord == -2:
        ret = tf.reduce_min(
            tf.linalg.svd(x, compute_uv=False), axis=axis, keepdims=keepdims
        )
    elif ord == "nuc":
        if tf.size(x).numpy() == 0:
            ret = x
        else:
            ret = tf.reduce_sum(tf.linalg.svd(x, compute_uv=False), axis=-1)
    else:
        ret = tf.linalg.norm(x, ord, axis, keepdims)

    if keepdims:
        ret = tf.reshape(ret, x.shape[:-2] + (1, 1))
    else:
        ret = tf.reshape(ret, x.shape[:-2])
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def matrix_power(
    x: Union[tf.Tensor, tf.Variable],
    n: int,
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
# noinspection PyPep8Naming
def matrix_rank(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    def dim_reduction(array):
        if array.ndim == 1:
            ret = array[0]
        elif array.ndim == 2:
            ret = array[0][0]
        elif array.ndim == 3:
            ret = array[0][0][0]
        elif array.ndim == 4:
            ret = array[0][0][0][0]
        return ret

    if len(x.shape) == 3:
        if x.shape[-3] == 0:
            return tf.constant(0, dtype=x.dtype)
    elif len(x.shape) > 3:
        if x.shape[-3] == 0 or x.shape[-4] == 0:
            return tf.constant(0, dtype=x.dtype)
    axis = None
    ret_shape = x.shape[:-2]
    if len(x.shape) == 2:
        singular_values = tf.linalg.svd(x, full_matrices=False, compute_uv=False)
    elif len(x.shape) > 2:
        y = tf.reshape(x, (-1, *x.shape[-2:]))
        singular_values = tf.stack(
            [
                tf.linalg.svd(split[0], full_matrices=False, compute_uv=False)
                for split in tf.split(y, y.shape[0], axis=0)
            ]
        )
        axis = 1
    if len(x.shape) < 2 or len(singular_values.shape) == 0:
        return tf.constant(0, dtype=x.dtype)
    max_values = tf.math.reduce_max(singular_values, axis=axis)
    if atol is None:
        if rtol is None:
            ret = ivy.sum(singular_values != 0, axis=axis)
        else:
            try:
                max_rtol = tf.cast(max_values, dtype=tf.float32) * tf.cast(
                    rtol, dtype=tf.float32
                )
            except ValueError:
                if ivy.all(
                    element == rtol[0] for element in rtol
                ):  # all elements are same in rtol
                    rtol = dim_reduction(rtol)
                    max_rtol = tf.cast(max_values, dtype=tf.float32) * tf.cast(
                        rtol, dtype=tf.float32
                    )
            if not isinstance(rtol, float) and tf.size(rtol) > 1:
                if ivy.all(
                    tf.math.equal(
                        max_rtol, tf.fill(max_rtol.shape, dim_reduction(max_rtol))
                    )
                ):
                    max_rtol = dim_reduction(max_rtol)
            elif not isinstance(max_values, float) and tf.size(max_values) > 1:
                if ivy.all(
                    tf.math.equal(max_values, tf.fill(max_values.shape, max_values[0]))
                ):
                    max_rtol = dim_reduction(max_rtol)
            ret = ivy.sum(
                tf.cast(singular_values, dtype=tf.float32)
                > tf.cast(max_rtol, dtype=tf.float32),
                axis=axis,
            )
    else:  # atol is not None
        if rtol is None:  # atol is not None, rtol is None
            ret = ivy.sum(singular_values > atol, axis=axis)
        else:
            tol = tf.experimental.numpy.max(atol, max_values * rtol)
            ret = ivy.sum(singular_values > tol, axis=axis)
    if len(ret_shape):
        ret = ivy.reshape(ret, ret_shape)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
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
    },
    backend_version,
)
def matrix_transpose(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.linalg.matrix_transpose(x)


# noinspection PyUnusedLocal,PyShadowingBuiltins
@with_unsupported_dtypes({"2.9.1 and below": ("complex", )}, backend_version)
def outer(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.outer(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
def pinv(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if rtol is None:
        ret = tf.linalg.pinv(x)
    else:
        x, rtol = ivy.promote_types_of_inputs(x, rtol)
        ret = tf.linalg.pinv(x, rtol)
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def qr(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    mode: str = "reduced",
    out: Optional[tf.Tensor] = None,
) -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    if mode == "reduced":
        q, r = tf.linalg.qr(x, full_matrices=False)
        ret = res(q, r)
    elif mode == "complete":
        q, r = tf.linalg.qr(x, full_matrices=True)
        ret = res(q, r)
    else:
        raise ivy.exceptions.IvyException(
            "Only 'reduced' and 'complete' qr modes are allowed "
            "for the tensorflow backend."
        )
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def slogdet(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    results = NamedTuple("slogdet", [("sign", tf.Tensor), ("logabsdet", tf.Tensor)])
    sign, logabsdet = tf.linalg.slogdet(x)
    return results(sign, logabsdet)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
def solve(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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
        if tf.math.reduce_any(tf.linalg.det(x1) == 0) or (
            x2.shape[-1] == x2.shape[-2] and tf.math.reduce_any(tf.linalg.det(x2) == 0)
        ):
            return x1
        ret = tf.linalg.solve(x1, x2)

    if expanded_last:
        ret = tf.squeeze(ret, axis=-1)
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
def svd(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Union[Union[tf.Tensor, tf.Variable], Tuple[Union[tf.Tensor, tf.Variable], ...]]:

    if compute_uv:
        results = namedtuple("svd", "U S Vh")

        batch_shape = tf.shape(x)[:-2]
        num_batch_dims = len(batch_shape)
        transpose_dims = list(range(num_batch_dims)) + [
            num_batch_dims + 1,
            num_batch_dims,
        ]
        D, U, V = tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        VT = tf.transpose(V, transpose_dims)
        return results(U, D, VT)
    else:
        results = namedtuple("svd", "S")
        D = tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(D)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def svdvals(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.linalg.svd(x, compute_uv=False)
    return ret


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def tensordot(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # find type to promote to
    dtype = ivy.as_native_dtype(ivy.promote_types(x1.dtype, x2.dtype))

    # type casting to float32 which is acceptable for tf.tensordot
    x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)

    ret = tf.cast(tf.tensordot(x1, x2, axes=axes), dtype)
    return ret


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
def trace(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not isinstance(x, tf.Variable):
        if len(x) == 0:
            return ivy.array([])
    return tf.experimental.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16", "float16", "complex")}, backend_version)
def vecdot(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(ivy.promote_types(x1.dtype, x2.dtype))
    if dtype != "float64":
        x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)
    else:
        x1, x2 = tf.cast(x1, tf.float64), tf.cast(x2, tf.float64)
    return tf.cast(tf.tensordot(x1, x2, axes=(axis, axis)), dtype)


@with_unsupported_dtypes({"2.9.1 and below": ("float16",)}, backend_version)
def vector_norm(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ord == -float("inf"):
        tn_normalized_vector = tf.reduce_min(tf.abs(x), axis, keepdims)
    elif ord == -2:
        tn_normalized_vector = 1.0 / tf.sqrt(
            tf.reduce_sum(1.0 / tf.abs(x) ** 2, axis, keepdims)
        )
    elif ord == -1:
        tn_normalized_vector = 1.0 / tf.reduce_sum(1.0 / tf.abs(x), axis, keepdims)
    elif ord == 0:
        tn_normalized_vector = tf.reduce_sum(tf.cast(x != 0, x.dtype), axis, keepdims)
    elif ord < 1:
        tn_normalized_vector = tf.reduce_sum(tf.abs(x) ** ord) ** (1.0 / ord)
    else:
        tn_normalized_vector = tf.linalg.norm(x, ord, axis, keepdims)
    if tn_normalized_vector.shape == tuple():
        ret = tf.expand_dims(tn_normalized_vector, 0)
    else:
        ret = tn_normalized_vector
    return ret


# Extra #
# ----- #


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def diag(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.diag(x, k=k)


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16", "complex")}, backend_version)
def vander(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.vander(x, N=N, increasing=increasing)


@with_unsupported_dtypes(
    {
        "2.9.1": (
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
            "complex"
        )
    },
    backend_version,
)
def vector_to_skew_symmetric_matrix(
    vector: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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
