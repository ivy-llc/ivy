# global
from hypothesis import strategies as st

# local
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# unravel_index
@st.composite
def max_value_as_shape_prod(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    dtype_and_x = draw(
        helpers.dtype_values_axis(
            available_dtypes=["int32", "int64"],
            min_value=0,
            max_value=np.prod(shape) - 1,
            min_num_dims=1,
        )
    )
    return dtype_and_x, shape


@handle_test(
    fn_tree="functional.ivy.experimental.unravel_index",
    dtype_x_shape=max_value_as_shape_prod(),
    test_gradients=st.just(False),
)
def test_unravel_index(*, dtype_x_shape, test_flags, backend_fw, fn_name, on_device):
    dtype_and_x, shape = dtype_x_shape
    input_dtype, x = dtype_and_x[0], dtype_and_x[1]
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        indices=np.asarray(x[0], dtype=input_dtype[0]),
        shape=shape,
    )
