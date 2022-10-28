# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# diagflat
@handle_test(
    fn_tree="functional.extensions.diagflat",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=50,
    ),
    k=helpers.ints(min_value=-49, max_value=49),
)
def test_diagflat(
    *,
    dtype_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    k,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
    )
