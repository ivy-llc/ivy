# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# ones
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ones"
    ),
)
def test_numpy_ones(
    shape,
    dtype,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_name="ones",
        shape=shape,
        dtype=dtype,
    )
