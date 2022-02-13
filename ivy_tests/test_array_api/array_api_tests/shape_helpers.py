import math
from itertools import product
from typing import Iterator, List, Optional, Tuple, Union

from ndindex import iter_indices as _iter_indices

from .typing import AtomicIndex, Index, Scalar, Shape

__all__ = [
    "broadcast_shapes",
    "normalise_axis",
    "ndindex",
    "axis_ndindex",
    "axes_ndindex",
    "reshape",
    "fmt_idx",
]


class BroadcastError(ValueError):
    """Shapes do not broadcast with eachother"""


def _broadcast_shapes(shape1: Shape, shape2: Shape) -> Shape:
    """Broadcasts `shape1` and `shape2`"""
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise BroadcastError()

        i = i - 1

    return tuple(shape)


def broadcast_shapes(*shapes: Shape):
    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result


def normalise_axis(
    axis: Optional[Union[int, Tuple[int, ...]]], ndim: int
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    return axes


def ndindex(shape: Shape) -> Iterator[Index]:
    """Yield every index of a shape"""
    return (indices[0] for indices in iter_indices(shape))


def iter_indices(
    *shapes: Shape, skip_axes: Tuple[int, ...] = ()
) -> Iterator[Tuple[Index, ...]]:
    """Wrapper for ndindex.iter_indices()"""
    # Prevent iterations if any shape has 0-sides
    for shape in shapes:
        if 0 in shape:
            return
    for indices in _iter_indices(*shapes, skip_axes=skip_axes):
        yield tuple(i.raw for i in indices)  # type: ignore


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


def reshape(flat_seq: List[Scalar], shape: Shape) -> Union[Scalar, List]:
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


def fmt_i(i: AtomicIndex) -> str:
    if isinstance(i, int):
        return str(i)
    elif isinstance(i, slice):
        res = ""
        if i.start is not None:
            res += str(i.start)
        res += ":"
        if i.stop is not None:
            res += str(i.stop)
        if i.step is not None:
            res += f":{i.step}"
        return res
    else:
        return "..."


def fmt_idx(sym: str, idx: Index) -> str:
    if idx == ():
        return sym
    res = f"{sym}["
    _idx = idx if isinstance(idx, tuple) else (idx,)
    if len(_idx) == 1:
        res += fmt_i(_idx[0])
    else:
        res += ", ".join(fmt_i(i) for i in _idx)
    res += "]"
    return res
