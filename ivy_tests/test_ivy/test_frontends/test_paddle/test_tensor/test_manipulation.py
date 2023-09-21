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
