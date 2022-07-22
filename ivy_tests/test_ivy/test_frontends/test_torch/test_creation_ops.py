# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.torch as ivy_torch


# ones
@given(
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_torch.valid_numeric_dtypes),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ones"
    ),
)
def test_torch_ones(
    size,
    dtype,
    with_out,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_name="ones",
        size=size,
    )
