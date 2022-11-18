# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# split
@handle_frontend_test(
    fn_tree="numpy.split",
    dtype_and_x_and_y_and_z=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_numpy_split(
    as_variable,
    dtype_and_x_and_y_and_z,
    num_positional_args,
):
    indices_or_sections, ary, axis, input_dtypes = dtype_and_x_and_y_and_z
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
