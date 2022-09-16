
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, given


# random
@handle_cmd_line_args
@given(
    input_dtypes=helpers.get_dtypes('integer', full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.random"
    ),
    size=helpers.get_shape(allow_none=True),
)
def test_numpy_random(
    input_dtypes,
    num_positional_args,
    size,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        test_values=False,
        fw=fw,
        frontend="numpy",
        fn_tree="random.random",
        size=size
    )
