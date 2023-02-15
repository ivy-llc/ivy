# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _get_dtype_and_matrix(draw):
    arbitrary_dims = draw(helpers.get_shape(max_dim_size=5))
    random_size = draw(st.integers(min_value=1, max_value=4))
    shape = (*arbitrary_dims, random_size, random_size)
    return draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )


@handle_cmd_line_args
@given(
    dtype_and_input=_get_dtype_and_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.bincount"
    ),
)
def test_tensorflow_det(
    dtype_and_input, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="bincount",
        weights=None,
        input=x[0],
        minlength=[0],
    )

