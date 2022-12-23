# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# as_dtype
@handle_frontend_test(
    fn_tree="tensorflow.as_dtype", input_dtype=helpers.get_dtypes("valid", full=False)
)
def test_tensorflow_as_dtype(
    *,
    input_dtype,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        type_value=input_dtype[0],
    )


# cast
@handle_frontend_test(
    fn_tree="tensorflow.cast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid"),
)
def test_tensorflow_cast(
    *,
    dtype_and_x,
    dtype,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype + dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dtype=dtype[0],
    )
