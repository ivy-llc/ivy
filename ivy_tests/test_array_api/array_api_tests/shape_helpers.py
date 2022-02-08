import math
from itertools import product
from typing import Iterator, List, Optional, Tuple, Union

from .typing import Scalar, Shape

__all__ = ["normalise_axis", "ndindex", "axis_ndindex", "axes_ndindex", "reshape"]


def normalise_axis(
    axis: Optional[Union[int, Tuple[int, ...]]], ndim: int
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    return axes


def ndindex(shape):
    """Iterator of n-D indices to an array

    Yields tuples of integers to index every element of an array of shape
    `shape`. Same as np.ndindex().
    """
    return product(*[range(i) for i in shape])


def axis_ndindex(
    shape: Shape, axis: int
) -> Iterator[Tuple[Tuple[Union[int, slice], ...], ...]]:
    """Generate indices that index all elements in dimensions beyond `axis`"""
    assert axis >= 0  # sanity check
    axis_indices = [range(side) for side in shape[:axis]]
    for _ in range(axis, len(shape)):
        axis_indices.append([slice(None, None)])
    yield from product(*axis_indices)


def axes_ndindex(shape: Shape, axes: Tuple[int, ...]) -> Iterator[List[Shape]]:
    """Generate indices that index all elements except in `axes` dimensions"""
    base_indices = []
    axes_indices = []
    for axis, side in enumerate(shape):
        if axis in axes:
            base_indices.append([None])
            axes_indices.append(range(side))
        else:
            base_indices.append(range(side))
            axes_indices.append([None])
    for base_idx in product(*base_indices):
        indices = []
        for idx in product(*axes_indices):
            idx = list(idx)
            for axis, side in enumerate(idx):
                if axis not in axes:
                    idx[axis] = base_idx[axis]
            idx = tuple(idx)
            indices.append(idx)
        yield list(indices)


def reshape(flat_seq: List[Scalar], shape: Shape) -> Union[Scalar, List[Scalar]]:
    """Reshape a flat sequence"""
    if any(s == 0 for s in shape):
        raise ValueError(
            f"{shape=} contains 0-sided dimensions, "
            f"but that's not representable in lists"
        )
    if len(shape) == 0:
        assert len(flat_seq) == 1  # sanity check
        return flat_seq[0]
    elif len(shape) == 1:
        return flat_seq
    size = len(flat_seq)
    n = math.prod(shape[1:])
    return [reshape(flat_seq[i * n : (i + 1) * n], shape[1:]) for i in range(size // n)]
