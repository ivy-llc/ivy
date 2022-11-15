# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="torch.multinomial",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    num_samples=helpers.ints(),
    replace=st.booleans(),
)
def test_torch_multinomial(
    *,
    dtype_and_values,
    num_samples,
    replace,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    input = value[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        num_samples=num_samples,
        replacement=replace,
        dtype=input_dtype,
    )


@handle_frontend_test(
    fn_tree="torch.manual_seed",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1, max_num_dims=1, min_dim_size=1, max_dim_size=1
            ),
            key="shape",
        ),
        min_value=0,
        max_value=2**32 - 1,
    ),
)
def test_torch_manual_seed(
    *,
    dtype_and_values,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        test_values=False,
        seed=value[0][0],
    )
