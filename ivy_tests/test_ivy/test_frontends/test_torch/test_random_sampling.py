# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    num_samples=helpers.ints(),
    replace=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.multinomial"
    ),
)
def test_torch_multinomial(
    dtype_and_values,
    num_samples,
    replace,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, value = dtype_and_values
    input = value[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=[False],
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="multinomial",
        input=input,
        num_samples=num_samples,
        replacement=replace,
        dtype=input_dtype,
    )
