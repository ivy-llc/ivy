# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Literal, List, NamedTuple
from collections import namedtuple

# local
import ivy
from ivy import inf
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# -------------------#


def cholesky(
    x: JaxArray, upper: bool = False, *, out: Optional[JaxArray] = None
) -> JaxArray:
    if not upper:
        ret = jnp.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = jnp.transpose(jnp.linalg.cholesky(jnp.transpose(x, axes=axes)), axes=axes)
    return ret


cholesky.unsupported_dtypes = ("float16",)


def cross(
    x1: JaxArray, x2: JaxArray, axis: int = -1, *, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.cross(a=x1, b=x2, axis=axis)
    return ret


def det(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.linalg.det(x)
    return ret


det.unsupported_dtypes = ("float16",)


def diagonal(
    x: JaxArray,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if not x.dtype == bool and not jnp.issubdtype(x.dtype, jnp.integer):
        ret = jnp.diagonal(x, offset, axis1, axis2)
        ret_edited = jnp.diagonal(
            x.at[1 / x == -jnp.inf].set(-jnp.inf), offset, axis1, axis2
        )
        ret_edited = ret_edited.at[ret_edited == -jnp.inf].set(-0.0)
        ret = ret.at[ret == ret_edited].set(ret_edited[ret == ret_edited])
    else:
        ret = jnp.diagonal(x, offset, axis1, axis2)
    return ret


def eigh(x: JaxArray) -> JaxArray:
    ret = jnp.linalg.eigh(x)
    return ret


eigh.unsupported_dtypes = ("float16",)


def eigvalsh(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.linalg.eigvalsh(x)
    return ret


eigvalsh.unsupported_dtypes = ("float16",)


def inv(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    if jnp.any(jnp.linalg.det(x.astype("float64")) == 0):
        ret = x
    else:
        ret = jnp.linalg.inv(x)
    return ret


inv.unsupported_dtypes = ("float16",)


def matmul(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.matmul(x1, x2)
    return ret


def matrix_norm(
    x: JaxArray,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if x.size == 0:
        if keepdims:
            ret = x.reshape(x.shape[:-2] + (1, 1))
        else:
            ret = x.reshape(x.shape[:-2])
    else:
        ret = jnp.linalg.norm(x, ord, (-2, -1), keepdims)
    return ret


matrix_norm.unsupported_dtypes = ("float16",)


def matrix_power(x: JaxArray, n: int, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.matrix_power(x, n)


def matrix_rank(
    x: JaxArray,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if x.size == 0:
        ret = 0
    elif x.size == 1:
        ret = jnp.count_nonzero(x)
    else:
        if x.ndim > 2:
        ret = jnp.linalg.matrix_rank(x, rtol)
    ret = jnp.asarray(ret, dtype=ivy.default_int_dtype(as_native=True))
    return ret


def matrix_transpose(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.swapaxes(x, -1, -2)
    return ret


matrix_transpose.unsupported_dtypes = ("float16", "int8")


def outer(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.outer(x1, x2)


def pinv(
    x: JaxArray,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if rtol is None:
        ret = jnp.linalg.pinv(x)
    else:
        ret = jnp.linalg.pinv(x, rtol)
    return ret


pinv.unsupported_dtypes = ("float16",)


def qr(x: JaxArray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = jnp.linalg.qr(x, mode=mode)
    ret = res(q, r)
    return ret


qr.unsupported_dtypes = ("float16",)


def slogdet(
    x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[JaxArray] = None
) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = jnp.linalg.slogdet(x)
    ret = results(sign, logabsdet)
    return ret


slogdet.unsupported_dtypes = ("float16",)


def solve(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = jnp.expand_dims(x2, axis=1)

    # if any of the arrays are empty
    is_empty_x1 = x1.size == 0
    is_empty_x2 = x2.size == 0
    if is_empty_x1 or is_empty_x2:
        for i in range(len(x1.shape) - 2):
            x2 = jnp.expand_dims(x2, axis=0)
        output_shape = list(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        output_shape.append(x2.shape[-2])
        output_shape.append(x2.shape[-1])
        ret = jnp.array([]).reshape(output_shape)
    else:
        output_shape = tuple(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        x1 = jnp.broadcast_to(x1, output_shape + x1.shape[-2:])
        x2 = jnp.broadcast_to(x2, output_shape + x2.shape[-2:])
        ret = jnp.linalg.solve(x1, x2)

    if expanded_last:
        ret = jnp.squeeze(ret, axis=-1)
    ret = jnp.asarray(ret, dtype=x1.dtype)
    return ret


solve.unsupported_dtypes = ("float16",)


def svd(
    x: JaxArray, full_matrices: bool = True
) -> Union[JaxArray, Tuple[JaxArray, ...]]:
    results = namedtuple("svd", "U S Vh")
    U, D, VT = jnp.linalg.svd(x, full_matrices=full_matrices)
    ret = results(U, D, VT)
    return ret


svd.unsupported_dtypes = ("float16",)


def svdvals(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.linalg.svd(x, compute_uv=False)
    return ret


svdvals.unsupported_dtypes = ("float16",)


def tensordot(
    x1: JaxArray,
    x2: JaxArray,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.tensordot(x1, x2, axes)
    return ret


def trace(x: JaxArray, offset: int = 0, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype)


trace.unsupported_dtypes = ("float16",)


def vecdot(
    x1: JaxArray, x2: JaxArray, axis: int = -1, *, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.tensordot(x1, x2, (axis, axis))
    return ret


def vector_norm(
    x: JaxArray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), ord, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, ord, axis, keepdims)

    if jnp_normalized_vector.shape == ():
        ret = jnp.expand_dims(jnp_normalized_vector, 0)
    else:
        ret = jnp_normalized_vector
    return ret


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(
    vector: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = jnp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = jnp.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = jnp.concatenate((zs, -a3s, a2s), -1)
    row2 = jnp.concatenate((a3s, zs, -a1s), -1)
    row3 = jnp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    ret = jnp.concatenate((row1, row2, row3), -2)
    return ret
