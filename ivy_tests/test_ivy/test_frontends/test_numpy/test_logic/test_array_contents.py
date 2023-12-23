# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


# isin
@st.composite
def _isin_data_generation_helper(draw):
    dtype_and_x = helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    )
    return draw(dtype_and_x)


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="numpy.allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_allclose(
    *,
    dtype_and_x,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="numpy.isclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_isclose(
    *,
    dtype_and_x,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
        equal_nan=equal_nan,
    )


# isin
@handle_frontend_test(
    fn_tree="numpy.isin",
    assume_unique_and_dtype_and_x=_isin_data_generation_helper(),
    invert=st.booleans(),
)
def test_numpy_isin(
    *,
    assume_unique_and_dtype_and_x,
    invert,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x_and_dtype = assume_unique_and_dtype_and_x
    dtypes, values = x_and_dtype
    elements, test_elements = values
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        element=elements,
        test_elements=test_elements,
        invert=invert,
        backend_to_test=backend_fw,
    )


@handle_frontend_test(
    fn_tree="numpy.isneginf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    test_with_out=st.just(False),
)
def test_numpy_isneginf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.isposinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    test_with_out=st.just(False),
)
def test_numpy_isposinf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
