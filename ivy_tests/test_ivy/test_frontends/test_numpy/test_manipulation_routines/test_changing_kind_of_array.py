# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# asanyarraya

@handle_frontend_test(
    fn_tree="numpy.asanyarray",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_numpy_asanyarray(
    *,
    dtype_and_x,
    as_variable,
    like,
    num_positional_args,
    native_array,
    on_device,
    order,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        dtype=input_dtype[0],
        order=order,
        like=like,
    )
