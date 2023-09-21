# global
from hypothesis import assume, strategies as st, given
import numpy as np

# local
import ivy.functional.frontends.numpy as np_frontend
from ivy.functional.frontends.numpy.ufunc import (
    ufuncs,
)


# --- Helpers --- #
# --------------- #


# strategy to generate a ufunc from given list
@st.composite
def generate_ufunc(draw, ufuncs=ufuncs):
    return draw(st.sampled_from(ufuncs))


# identity
@given(
    ufunc_name=generate_ufunc(),
)
def test_numpy_identity(
    ufunc_name,
):
    assume(hasattr(np_frontend, ufunc_name))
    frontend_ufunc = getattr(np_frontend, ufunc_name)
    np_ufunc = getattr(np, ufunc_name)
    assert frontend_ufunc.identity == np_ufunc.identity


# nargs
@given(
    ufunc_name=generate_ufunc(),
)
def test_numpy_nargs(
    ufunc_name,
):
    assume(hasattr(np_frontend, ufunc_name))
    frontend_ufunc = getattr(np_frontend, ufunc_name)
    np_ufunc = getattr(np, ufunc_name)
    assert frontend_ufunc.nargs == np_ufunc.nargs


# nin
@given(
    ufunc_name=generate_ufunc(),
)
def test_numpy_nin(
    ufunc_name,
):
    assume(hasattr(np_frontend, ufunc_name))
    frontend_ufunc = getattr(np_frontend, ufunc_name)
    np_ufunc = getattr(np, ufunc_name)
    assert frontend_ufunc.nin == np_ufunc.nin


# nout
@given(
    ufunc_name=generate_ufunc(),
)
def test_numpy_nout(
    ufunc_name,
):
    assume(hasattr(np_frontend, ufunc_name))
    frontend_ufunc = getattr(np_frontend, ufunc_name)
    np_ufunc = getattr(np, ufunc_name)
    assert frontend_ufunc.nout == np_ufunc.nout
