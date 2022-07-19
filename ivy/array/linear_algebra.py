# global
import abc
from typing import Union, Optional, Literal, NamedTuple, Tuple, List

# local
import ivy

inf = float("inf")


class ArrayWithLinearAlgebra(abc.ABC):
    def matmul(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.matmul(self._data, x2, out=out)

    def cholesky(
        self: ivy.Array,
        upper: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.cholesky(self._data, upper, out=out)

    def cross(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.cross(self._data, x2, axis, out=out)

    def det(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.det(self._data, out=out)

    def diagonal(
        self: ivy.Array,
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.diagonal(self._data, offset, axis1, axis2, out=out)

    def eigh(
        self: ivy.Array,
    ) -> NamedTuple:
        return ivy.eigh(self._data)

    def eigvalsh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.eigvalsh(self._data, out=out)

    def inv(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.inv(self._data, out=out)

    def matrix_norm(
        self: ivy.Array,
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.matrix_norm(self._data, ord, keepdims, out=out)

    def matrix_rank(
        self: ivy.Array,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.matrix_rank(self._data, rtol, out=out)

    def matrix_transpose(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.matrix_transpose(self._data, out=out)

    def outer(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.outer(self._data, x2, out=out)

    def pinv(
        self: ivy.Array,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.pinv(self._data, rtol, out=out)

    def qr(
        self: ivy.Array,
        mode: str = "reduced",
    ) -> NamedTuple:
        return ivy.qr(self._data, mode)

    def solve(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.solve(self._data, x2, out=out)

    def svd(
        self: ivy.Array,
        full_matrices: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
        return ivy.svd(self._data, full_matrices, out=out)

    def svdvals(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.svdvals(self._data, out=out)

    def tensordot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.tensordot(self._data, x2, axes, out=out)

    def trace(
        self: ivy.Array,
        offset: int = 0,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.trace(self._data, offset, out=out)

    def vecdot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vecdot(self._data, x2, axis, out=out)

    def vector_norm(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        ord: Union[int, float, Literal[inf, -inf]] = 2,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vector_norm(self._data, axis, keepdims, ord, out=out)

    def vector_to_skew_symmetric_matrix(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.vector_to_skew_symmetric_matrix(self._data, out=out)
