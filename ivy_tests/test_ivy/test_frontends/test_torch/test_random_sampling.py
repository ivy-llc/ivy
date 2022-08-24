# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.torch as ivy_torch


@handle_cmd_line_args
@given(
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.seed"
    ),
)
def test_torch_seed(
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw
):
    helpers.test_frontend_function(
        input_dtypes=ivy_torch.valid_int_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="seed"
    )
