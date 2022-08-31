# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# reshape
@handle_cmd_line_args
@given(
    x_dtypes=helpers.dtype_and_values(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        )
    ),
    copy=st.booleans(),
    with_out=st.booleans(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.reshape"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_reshape(
    x_dtypes,
    copy,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtypes, x = x_dtypes
    shape = helpers.reshape_shapes(shape=x.shape)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="reshape",
        x=x,
        shape=shape,
        copy=copy,
    )
