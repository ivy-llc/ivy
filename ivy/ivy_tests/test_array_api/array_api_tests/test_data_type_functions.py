import struct
from typing import Union

import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import DataType

pytestmark = pytest.mark.ci


def float32(n: Union[int, float]) -> float:
    return struct.unpack("!f", struct.pack("!f", float(n)))[0]


@given(
    x_dtype=xps.scalar_dtypes(),
    dtype=xps.scalar_dtypes(),
    kw=hh.kwargs(copy=st.booleans()),
    data=st.data(),
)
def test_astype(x_dtype, dtype, kw, data):
    if xp.bool in (x_dtype, dtype):
        elements_strat = xps.from_dtype(x_dtype)
    else:
        m1, M1 = dh.dtype_ranges[x_dtype]
        m2, M2 = dh.dtype_ranges[dtype]
        if dh.is_int_dtype(x_dtype):
            cast = int
        elif x_dtype == xp.float32:
            cast = float32
        else:
            cast = float
        min_value = cast(max(m1, m2))
        max_value = cast(min(M1, M2))
        elements_strat = xps.from_dtype(
            x_dtype,
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    x = data.draw(
        xps.arrays(dtype=x_dtype, shape=hh.shapes(), elements=elements_strat), label="x"
    )

    out = xp.astype(x, dtype, **kw)

    ph.assert_kw_dtype("astype", dtype, out.dtype)
    ph.assert_shape("astype", out.shape, x.shape)
    # TODO: test values
    # TODO: test copy


@given(
    shapes=st.integers(1, 5).flatmap(hh.mutually_broadcastable_shapes), data=st.data()
)
def test_broadcast_arrays(shapes, data):
    arrays = []
    for c, shape in enumerate(shapes, 1):
        x = data.draw(xps.arrays(dtype=xps.scalar_dtypes(), shape=shape), label=f"x{c}")
        arrays.append(x)

    out = xp.broadcast_arrays(*arrays)

    out_shape = sh.broadcast_shapes(*shapes)
    for i, x in enumerate(arrays):
        ph.assert_dtype(
            "broadcast_arrays", x.dtype, out[i].dtype, repr_name=f"out[{i}].dtype"
        )
        ph.assert_result_shape(
            "broadcast_arrays",
            shapes,
            out[i].shape,
            out_shape,
            repr_name=f"out[{i}].shape",
        )
    # TODO: test values


@given(x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()), data=st.data())
def test_broadcast_to(x, data):
    shape = data.draw(
        hh.mutually_broadcastable_shapes(1, base_shape=x.shape)
        .map(lambda S: S[0])
        .filter(lambda s: sh.broadcast_shapes(x.shape, s) == s),
        label="shape",
    )

    out = xp.broadcast_to(x, shape)

    ph.assert_dtype("broadcast_to", x.dtype, out.dtype)
    ph.assert_shape("broadcast_to", out.shape, shape)
    # TODO: test values


@given(_from=xps.scalar_dtypes(), to=xps.scalar_dtypes(), data=st.data())
def test_can_cast(_from, to, data):
    from_ = data.draw(
        st.just(_from) | xps.arrays(dtype=_from, shape=hh.shapes()), label="from_"
    )

    out = xp.can_cast(from_, to)

    f_func = f"[can_cast({dh.dtype_to_name[_from]}, {dh.dtype_to_name[to]})]"
    assert isinstance(out, bool), f"{type(out)=}, but should be bool {f_func}"
    if _from == xp.bool:
        expected = to == xp.bool
    else:
        for dtypes in [dh.all_int_dtypes, dh.float_dtypes]:
            if _from in dtypes:
                same_family = to in dtypes
                break
        if same_family:
            from_min, from_max = dh.dtype_ranges[_from]
            to_min, to_max = dh.dtype_ranges[to]
            expected = from_min >= to_min and from_max <= to_max
        else:
            expected = False
    assert out == expected, f"{out=}, but should be {expected} {f_func}"


def make_dtype_id(dtype: DataType) -> str:
    return dh.dtype_to_name[dtype]


@pytest.mark.parametrize("dtype", dh.float_dtypes, ids=make_dtype_id)
def test_finfo(dtype):
    out = xp.finfo(dtype)
    f_func = f"[finfo({dh.dtype_to_name[dtype]})]"
    for attr, stype in [
        ("bits", int),
        ("eps", float),
        ("max", float),
        ("min", float),
        ("smallest_normal", float),
    ]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, stype
        ), f"type(out.{attr})={type(value)!r}, but should be {stype.__name__} {f_func}"
    # TODO: test values


@pytest.mark.parametrize("dtype", dh.all_int_dtypes, ids=make_dtype_id)
def test_iinfo(dtype):
    out = xp.iinfo(dtype)
    f_func = f"[iinfo({dh.dtype_to_name[dtype]})]"
    for attr in ["bits", "max", "min"]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, int
        ), f"type(out.{attr})={type(value)!r}, but should be int {f_func}"
    # TODO: test values


@given(hh.mutually_promotable_dtypes(None))
def test_result_type(dtypes):
    out = xp.result_type(*dtypes)
    ph.assert_dtype("result_type", dtypes, out, repr_name="out")
