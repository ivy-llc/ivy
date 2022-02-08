import math
from collections import deque
from typing import Iterable, Iterator, Tuple, Union

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import Array, Shape

pytestmark = pytest.mark.ci

MAX_SIDE = hh.MAX_ARRAY_SIZE // 64
MAX_DIMS = min(hh.MAX_ARRAY_SIZE // MAX_SIDE, 32)  # NumPy only supports up to 32 dims


def shared_shapes(*args, **kwargs) -> st.SearchStrategy[Shape]:
    key = "shape"
    if args:
        key += " " + " ".join(args)
    if kwargs:
        key += " " + ph.fmt_kw(kwargs)
    return st.shared(hh.shapes(*args, **kwargs), key="shape")


def assert_array_ndindex(
    func_name: str,
    x: Array,
    x_indices: Iterable[Union[int, Shape]],
    out: Array,
    out_indices: Iterable[Union[int, Shape]],
    /,
    **kw,
):
    msg_suffix = f" [{func_name}({ph.fmt_kw(kw)})]\n  {x=}\n{out=}"
    for x_idx, out_idx in zip(x_indices, out_indices):
        msg = f"out[{out_idx}]={out[out_idx]}, should be x[{x_idx}]={x[x_idx]}"
        msg += msg_suffix
        if dh.is_float_dtype(x.dtype) and xp.isnan(x[x_idx]):
            assert xp.isnan(out[out_idx]), msg
        else:
            assert out[out_idx] == x[x_idx], msg


@given(
    dtypes=hh.mutually_promotable_dtypes(None, dtypes=dh.numeric_dtypes),
    base_shape=hh.shapes(),
    data=st.data(),
)
def test_concat(dtypes, base_shape, data):
    axis_strat = st.none()
    ndim = len(base_shape)
    if ndim > 0:
        axis_strat |= st.integers(-ndim, ndim - 1)
    kw = data.draw(
        axis_strat.flatmap(lambda a: hh.specified_kwargs(("axis", a, 0))), label="kw"
    )
    axis = kw.get("axis", 0)
    if axis is None:
        _axis = None
        shape_strat = hh.shapes()
    else:
        _axis = axis if axis >= 0 else len(base_shape) + axis
        shape_strat = st.integers(0, MAX_SIDE).map(
            lambda i: base_shape[:_axis] + (i,) + base_shape[_axis + 1 :]
        )
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape_strat), label=f"x{i}")
        arrays.append(x)

    out = xp.concat(arrays, **kw)

    ph.assert_dtype("concat", dtypes, out.dtype)

    shapes = tuple(x.shape for x in arrays)
    if _axis is None:
        size = sum(math.prod(s) for s in shapes)
        shape = (size,)
    else:
        shape = list(shapes[0])
        for other_shape in shapes[1:]:
            shape[_axis] += other_shape[_axis]
        shape = tuple(shape)
    ph.assert_result_shape("concat", shapes, out.shape, shape, **kw)

    if _axis is None:
        out_indices = (i for i in range(out.size))
        for x_num, x in enumerate(arrays, 1):
            for x_idx in sh.ndindex(x.shape):
                out_i = next(out_indices)
                ph.assert_0d_equals(
                    "concat",
                    f"x{x_num}[{x_idx}]",
                    x[x_idx],
                    f"out[{out_i}]",
                    out[out_i],
                    **kw,
                )
    else:
        out_indices = sh.ndindex(out.shape)
        for idx in sh.axis_ndindex(shapes[0], _axis):
            f_idx = ", ".join(str(i) if isinstance(i, int) else ":" for i in idx)
            for x_num, x in enumerate(arrays, 1):
                indexed_x = x[idx]
                for x_idx in sh.ndindex(indexed_x.shape):
                    out_idx = next(out_indices)
                    ph.assert_0d_equals(
                        "concat",
                        f"x{x_num}[{f_idx}][{x_idx}]",
                        indexed_x[x_idx],
                        f"out[{out_idx}]",
                        out[out_idx],
                        **kw,
                    )


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes()),
    axis=shared_shapes().flatmap(
        # Generate both valid and invalid axis
        lambda s: st.integers(2 * (-len(s) - 1), 2 * len(s))
    ),
)
def test_expand_dims(x, axis):
    if axis < -x.ndim - 1 or axis > x.ndim:
        with pytest.raises(IndexError):
            xp.expand_dims(x, axis=axis)
        return

    out = xp.expand_dims(x, axis=axis)

    ph.assert_dtype("expand_dims", x.dtype, out.dtype)

    shape = [side for side in x.shape]
    index = axis if axis >= 0 else x.ndim + axis + 1
    shape.insert(index, 1)
    shape = tuple(shape)
    ph.assert_result_shape("expand_dims", (x.shape,), out.shape, shape)

    assert_array_ndindex(
        "expand_dims", x, sh.ndindex(x.shape), out, sh.ndindex(out.shape)
    )


@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1).filter(lambda s: 1 in s)
    ),
    data=st.data(),
)
def test_squeeze(x, data):
    axes = st.integers(-x.ndim, x.ndim - 1)
    axis = data.draw(
        axes
        | st.lists(axes, unique_by=lambda i: i if i >= 0 else i + x.ndim).map(tuple),
        label="axis",
    )

    axes = (axis,) if isinstance(axis, int) else axis
    axes = sh.normalise_axis(axes, x.ndim)

    squeezable_axes = [i for i, side in enumerate(x.shape) if side == 1]
    if any(i not in squeezable_axes for i in axes):
        with pytest.raises(ValueError):
            xp.squeeze(x, axis)
        return

    out = xp.squeeze(x, axis)

    ph.assert_dtype("squeeze", x.dtype, out.dtype)

    shape = []
    for i, side in enumerate(x.shape):
        if i not in axes:
            shape.append(side)
    shape = tuple(shape)
    ph.assert_result_shape("squeeze", (x.shape,), out.shape, shape, axis=axis)

    assert_array_ndindex("squeeze", x, sh.ndindex(x.shape), out, sh.ndindex(out.shape))


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    data=st.data(),
)
def test_flip(x, data):
    if x.ndim == 0:
        axis_strat = st.none()
    else:
        axis_strat = (
            st.none() | st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim)
        )
    kw = data.draw(hh.kwargs(axis=axis_strat), label="kw")

    out = xp.flip(x, **kw)

    ph.assert_dtype("flip", x.dtype, out.dtype)

    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    for indices in sh.axes_ndindex(x.shape, _axes):
        reverse_indices = indices[::-1]
        assert_array_ndindex("flip", x, indices, out, reverse_indices)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes(min_dims=1)),
    axes=shared_shapes(min_dims=1).flatmap(
        lambda s: st.lists(
            st.integers(0, len(s) - 1),
            min_size=len(s),
            max_size=len(s),
            unique=True,
        ).map(tuple)
    ),
)
def test_permute_dims(x, axes):
    out = xp.permute_dims(x, axes)

    ph.assert_dtype("permute_dims", x.dtype, out.dtype)

    shape = [None for _ in range(len(axes))]
    for i, dim in enumerate(axes):
        side = x.shape[dim]
        shape[i] = side
    shape = tuple(shape)
    ph.assert_result_shape("permute_dims", (x.shape,), out.shape, shape, axes=axes)

    indices = list(sh.ndindex(x.shape))
    permuted_indices = [tuple(idx[axis] for axis in axes) for idx in indices]
    assert_array_ndindex("permute_dims", x, indices, out, permuted_indices)


@st.composite
def reshape_shapes(draw, shape):
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(st.integers(0)).filter(lambda s: math.prod(s) == size))
    assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(st.integers(0, len(rshape) - 1))
        rshape[index] = -1
    return tuple(rshape)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(max_side=MAX_SIDE)),
    data=st.data(),
)
def test_reshape(x, data):
    shape = data.draw(reshape_shapes(x.shape))

    out = xp.reshape(x, shape)

    ph.assert_dtype("reshape", x.dtype, out.dtype)

    _shape = list(shape)
    if any(side == -1 for side in shape):
        size = math.prod(x.shape)
        rsize = math.prod(shape) * -1
        _shape[shape.index(-1)] = size / rsize
    _shape = tuple(_shape)
    ph.assert_result_shape("reshape", (x.shape,), out.shape, _shape, shape=shape)

    assert_array_ndindex("reshape", x, sh.ndindex(x.shape), out, sh.ndindex(out.shape))


def roll_ndindex(shape: Shape, shifts: Tuple[int], axes: Tuple[int]) -> Iterator[Shape]:
    assert len(shifts) == len(axes)  # sanity check
    all_shifts = [0 for _ in shape]
    for s, a in zip(shifts, axes):
        all_shifts[a] = s
    for idx in sh.ndindex(shape):
        yield tuple((i + sh) % si for i, sh, si in zip(idx, all_shifts, shape))


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes()), st.data())
def test_roll(x, data):
    shift_strat = st.integers(-hh.MAX_ARRAY_SIZE, hh.MAX_ARRAY_SIZE)
    if x.ndim > 0:
        shift_strat = shift_strat | st.lists(
            shift_strat, min_size=1, max_size=x.ndim
        ).map(tuple)
    shift = data.draw(shift_strat, label="shift")
    if isinstance(shift, tuple):
        axis_strat = xps.valid_tuple_axes(x.ndim).filter(lambda t: len(t) == len(shift))
        kw_strat = axis_strat.map(lambda t: {"axis": t})
    else:
        axis_strat = st.none()
        if x.ndim != 0:
            axis_strat |= st.integers(-x.ndim, x.ndim - 1)
        kw_strat = hh.kwargs(axis=axis_strat)
    kw = data.draw(kw_strat, label="kw")

    out = xp.roll(x, shift, **kw)

    kw = {"shift": shift, **kw}  # for error messages

    ph.assert_dtype("roll", x.dtype, out.dtype)

    ph.assert_result_shape("roll", (x.shape,), out.shape)

    if kw.get("axis", None) is None:
        assert isinstance(shift, int)  # sanity check
        indices = list(sh.ndindex(x.shape))
        shifted_indices = deque(indices)
        shifted_indices.rotate(-shift)
        assert_array_ndindex("roll", x, indices, out, shifted_indices, **kw)
    else:
        shifts = (shift,) if isinstance(shift, int) else shift
        axes = sh.normalise_axis(kw["axis"], x.ndim)
        shifted_indices = roll_ndindex(x.shape, shifts, axes)
        assert_array_ndindex("roll", x, sh.ndindex(x.shape), out, shifted_indices, **kw)


@given(
    shape=shared_shapes(min_dims=1),
    dtypes=hh.mutually_promotable_dtypes(None),
    kw=hh.kwargs(
        axis=shared_shapes(min_dims=1).flatmap(
            lambda s: st.integers(-len(s), len(s) - 1)
        )
    ),
    data=st.data(),
)
def test_stack(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)

    out = xp.stack(arrays, **kw)

    ph.assert_dtype("stack", dtypes, out.dtype)

    axis = kw.get("axis", 0)
    _axis = axis if axis >= 0 else len(shape) + axis + 1
    _shape = list(shape)
    _shape.insert(_axis, len(arrays))
    _shape = tuple(_shape)
    ph.assert_result_shape(
        "stack", tuple(x.shape for x in arrays), out.shape, _shape, **kw
    )

    out_indices = sh.ndindex(out.shape)
    for idx in sh.axis_ndindex(arrays[0].shape, axis=_axis):
        f_idx = ", ".join(str(i) if isinstance(i, int) else ":" for i in idx)
        print(f"{f_idx=}")
        for x_num, x in enumerate(arrays, 1):
            indexed_x = x[idx]
            for x_idx in sh.ndindex(indexed_x.shape):
                out_idx = next(out_indices)
                ph.assert_0d_equals(
                    "stack",
                    f"x{x_num}[{f_idx}][{x_idx}]",
                    indexed_x[x_idx],
                    f"out[{out_idx}]",
                    out[out_idx],
                    **kw,
                )
