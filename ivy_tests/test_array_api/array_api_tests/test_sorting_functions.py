import math
import pytest
from typing import Set

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.control import assume

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import Scalar, Shape

pytestmark = pytest.mark.ci


def assert_scalar_in_set(
    func_name: str,
    idx: Shape,
    out: Scalar,
    set_: Set[Scalar],
    /,
    **kw,
):
    out_repr = "out" if idx == () else f"out[{idx}]"
    if math.isnan(out):
        raise NotImplementedError()
    msg = f"{out_repr}={out}, but should be in {set_} [{func_name}({ph.fmt_kw(kw)})]"
    assert out in set_, msg


# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argsort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.argsort(x, **kw)

    ph.assert_default_index("argsort", out.dtype)
    ph.assert_shape("argsort", out.shape, x.shape, **kw)
    axis = kw.get("axis", -1)
    axes = sh.normalise_axis(axis, x.ndim)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements = [scalar_type(x[idx]) for idx in indices]
        orders = list(range(len(elements)))
        sorders = sorted(
            orders, key=elements.__getitem__, reverse=kw.get("descending", False)
        )
        if kw.get("stable", True):
            for idx, o in zip(indices, sorders):
                ph.assert_scalar_equals("argsort", int, idx, int(out[idx]), o, **kw)
        else:
            idx_elements = dict(zip(indices, elements))
            idx_orders = dict(zip(indices, orders))
            element_orders = {}
            for e in set(elements):
                element_orders[e] = [
                    idx_orders[idx] for idx in indices if idx_elements[idx] == e
                ]
            selements = [elements[o] for o in sorders]
            for idx, e in zip(indices, selements):
                expected_orders = element_orders[e]
                out_o = int(out[idx])
                if len(expected_orders) == 1:
                    ph.assert_scalar_equals(
                        "argsort", int, idx, out_o, expected_orders[0], **kw
                    )
                else:
                    assert_scalar_in_set(
                        "argsort", idx, out_o, set(expected_orders), **kw
                    )


# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_sort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.sort(x, **kw)

    ph.assert_dtype("sort", out.dtype, x.dtype)
    ph.assert_shape("sort", out.shape, x.shape, **kw)
    axis = kw.get("axis", -1)
    axes = sh.normalise_axis(axis, x.ndim)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements = [scalar_type(x[idx]) for idx in indices]
        size = len(elements)
        orders = sorted(
            range(size), key=elements.__getitem__, reverse=kw.get("descending", False)
        )
        for out_idx, o in zip(indices, orders):
            x_idx = indices[o]
            # TODO: error message when unstable should not imply just one idx
            ph.assert_0d_equals(
                "sort",
                f"x[{x_idx}]",
                x[x_idx],
                f"out[{out_idx}]",
                out[out_idx],
                **kw,
            )
