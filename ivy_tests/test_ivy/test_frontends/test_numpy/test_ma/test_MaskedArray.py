# global
from hypothesis import strategies as st

# local
import ivy
from ivy.functional.frontends.numpy.ma.MaskedArray import MaskedArray
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _array_mask(draw):
    dtype = draw(helpers.get_dtypes("valid", prune_function=False, full=False))
    dtypes, x_mask = draw(
        helpers.dtype_and_values(
            num_arrays=2,
            dtype=[dtype[0], "bool"],
        )
    )
    return dtype[0], x_mask


# --- Main --- #
# ------------ #


# data
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_array_mask(),
)
def test_numpy_data(
    args,
):
    dtype, data = args
    x = MaskedArray(data[0], mask=data[1], dtype=dtype)
    assert ivy.all(x.data == ivy.array(data[0]))


# dtype
@handle_frontend_test(fn_tree="numpy.add", dtype_x_mask=_array_mask())  # dummy fn_tree
def test_numpy_dtype(dtype_x_mask):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype)
    assert x.dtype == dtype


# fill_value
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    dtype_x_mask=_array_mask(),
    fill=st.integers(),
)
def test_numpy_fill_value(
    dtype_x_mask,
    fill,
):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype, fill_value=fill)
    assert x.fill_value == ivy.array(fill, dtype=dtype)


# hardmask
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    dtype_x_mask=_array_mask(),
    hard=st.booleans(),
)
def test_numpy_hardmask(dtype_x_mask, hard):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype, hard_mask=hard)
    assert x.hardmask == hard


# mask
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_array_mask(),
)
def test_numpy_mask(args):
    dtype, data = args
    x = MaskedArray(data[0], mask=ivy.array(data[1]), dtype=dtype, shrink=False)
    assert ivy.all(x.mask == ivy.array(data[1]))
