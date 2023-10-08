# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    _get_dtype_values_k_axes_for_rot90,
)


# --- Helpers --- #
# --------------- #


@st.composite
def dtypes_x_reshape_(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    return dtypes, x, shape


@handle_frontend_test(
    fn_tree="paddle.tensor.manipulation.index_add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=3,
        num_arrays=3,
    ),
    dtype_and_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=2,
        num_arrays=1,
    ),
    dtype_and_x3=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=3,
        num_arrays=2,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_paddle_index_add(
    *,
    dtype_and_x,
    dtype_and_x2,
    dtype_and_x3,
    dtype,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    index = dtype_and_x2[1]
    value = dtype_and_x3[1]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        index=index[0],
        axis=0,
        value=value[0],
        dtype=dtype[0],
    )


# reshape_
@handle_frontend_test(
    fn_tree="paddle.tensor.manipulation.reshape_",
    dtypes_x_reshape=dtypes_x_reshape_(),
)
def test_paddle_reshape_(
    *,
    dtypes_x_reshape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )
