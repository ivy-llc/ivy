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
    dtype1=helpers.get_dtypes("numeric", full=False, none=True),
    dtype2=helpers.get_dtypes("numeric", full=False, none=True),
    dtype3=helpers.get_dtypes("numeric", full=False, none=True),
    dtype4=helpers.get_dtypes("numeric", full=False, none=True),
)
def test_numpy_split(
    as_variable,
    dtype1,
    dtype2,
    dtype3,
    dtype4,
    num_positional_args,
):
    indices_or_sections, ary, axis, input_dtypes = dtype1, dtype2, dtype3, dtype4
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
