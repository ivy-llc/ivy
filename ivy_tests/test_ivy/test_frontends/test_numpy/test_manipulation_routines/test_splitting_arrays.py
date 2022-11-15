from hypothesis import given
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# split
@handle_cmd_line_args
@given(
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.split"
    ),
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_numpy_split(
    as_variable,
    dtype,
    num_positional_args,
):
    indices_or_sections, ary, axis, input_dtypes = dtype
    helpers.test_frontend_function(
        as_variable_flags=as_variable,
        input_dtypes=input_dtypes,
        with_out=False,
        num_positional_args=num_positional_args,
        frontend="numpy",
        fn_tree="split",
        ary=ary[0],
        axis=axis,
        indices_or_sections=indices_or_sections[0],
    )
