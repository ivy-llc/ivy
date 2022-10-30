from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(fn_name="triu_indices"),
)
def test_triu_indices(
    *,
    n_rows,
    n_cols,
    k,
    device,
    num_positional_args,
    fw,
):
    helpers.test_function(
        input_dtypes=["int32"],
        as_variable_flags=[False],
        with_out=None,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        container_flags=[False],
        instance_method=False,
        fw=fw,
        fn_name="triu_indices",
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        device=device,
    )
