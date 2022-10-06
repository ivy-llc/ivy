# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# can_cast
@handle_cmd_line_args
@given(
    dtypes=helpers. get_dtypes("valid", full=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.can_cast"))
def test_torch_can_cast(
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
):

    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="can_cast",
        from_=dtypes[0],
        to=dtypes[1],
    )
