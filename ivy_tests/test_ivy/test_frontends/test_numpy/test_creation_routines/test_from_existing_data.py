# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

@handle_cmd_line_args
@given(
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.functional.frontends.numpy.asarray"),
)
def test_numpy_asarray(
    dtype_and_a,
    as_variable,
    num_positional_args,
    fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        with_out=False,
        fw=fw,
        frontend="numpy",
        fn_tree="asarray",
        a=a,
        dtype=dtype,
    )
