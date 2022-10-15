# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    population_size=helpers.ints(),
    num_samples=helpers.ints(),
    replace=helpers.bool_val_flags(),
    dtypes=helpers.get_dtypes("float", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.multinomial"
    ),
)
def test_torch_multinomial(
        population_size,
        num_samples,
        replace,
        out,
        dtypes,
        num_positional_args,
        native_array,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=[False],
        with_out=out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="multinomial",
        input=population_size,
        num_samples=num_samples,
        replacement=replace,
        dtype=dtypes,
    )
