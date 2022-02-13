import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps

pytestmark = pytest.mark.ci


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmax(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    out = xp.argmax(x, **kw)

    ph.assert_default_index("argmax", out.dtype)
    axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "argmax", out.shape, x.shape, axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
        max_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(range(len(elements)), key=elements.__getitem__)
        ph.assert_scalar_equals("argmax", int, out_idx, max_i, expected)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmin(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    out = xp.argmin(x, **kw)

    ph.assert_default_index("argmin", out.dtype)
    axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "argmin", out.shape, x.shape, axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
        min_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(range(len(elements)), key=elements.__getitem__)
        ph.assert_scalar_equals("argmin", int, out_idx, min_i, expected)


@pytest.mark.data_dependent_shapes
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_nonzero(x):
    out = xp.nonzero(x)
    if x.ndim == 0:
        assert len(out) == 1, f"{len(out)=}, but should be 1 for 0-dimensional arrays"
    else:
        assert len(out) == x.ndim, f"{len(out)=}, but should be {x.ndim=}"
    size = out[0].size
    for i in range(len(out)):
        assert out[i].ndim == 1, f"out[{i}].ndim={x.ndim}, but should be 1"
        assert (
            out[i].size == size
        ), f"out[{i}].size={x.size}, but should be out[0].size={size}"
        ph.assert_default_index("nonzero", out[i].dtype, repr_name=f"out[{i}].dtype")
    indices = []
    if x.dtype == xp.bool:
        for idx in sh.ndindex(x.shape):
            if x[idx]:
                indices.append(idx)
    else:
        for idx in sh.ndindex(x.shape):
            if x[idx] != 0:
                indices.append(idx)
    if x.ndim == 0:
        assert out[0].size == len(
            indices
        ), f"{out[0].size=}, but should be {len(indices)}"
    else:
        for i in range(size):
            idx = tuple(int(x[i]) for x in out)
            f_idx = f"Extrapolated index (x[{i}] for x in out)={idx}"
            f_element = f"x[{idx}]={x[idx]}"
            assert idx in indices, f"{f_idx} results in {f_element}, a zero element"
            assert (
                idx == indices[i]
            ), f"{f_idx} is in the wrong position, should be {indices.index(idx)}"


@given(
    shapes=hh.mutually_broadcastable_shapes(3),
    dtypes=hh.mutually_promotable_dtypes(),
    data=st.data(),
)
def test_where(shapes, dtypes, data):
    cond = data.draw(xps.arrays(dtype=xp.bool, shape=shapes[0]), label="condition")
    x1 = data.draw(xps.arrays(dtype=dtypes[0], shape=shapes[1]), label="x1")
    x2 = data.draw(xps.arrays(dtype=dtypes[1], shape=shapes[2]), label="x2")

    out = xp.where(cond, x1, x2)

    shape = sh.broadcast_shapes(*shapes)
    ph.assert_shape("where", out.shape, shape)
    # TODO: generate indices without broadcasting arrays
    _cond = xp.broadcast_to(cond, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)
    for idx in sh.ndindex(shape):
        if _cond[idx]:
            ph.assert_0d_equals(
                "where", f"_x1[{idx}]", _x1[idx], f"out[{idx}]", out[idx]
            )
        else:
            ph.assert_0d_equals(
                "where", f"_x2[{idx}]", _x2[idx], f"out[{idx}]", out[idx]
            )
