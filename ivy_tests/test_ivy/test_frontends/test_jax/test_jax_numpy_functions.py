# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.jax as ivy_jax


# ones
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_jax.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.ones"
    ),
)
def test_jax_numpy_ones(
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
        frontend="jax",
        fn_name="numpy.ones",
        shape=shape,
        dtype=dtype,
    )
