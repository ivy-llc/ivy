from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.extensions.triu_indices",
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    input_dtype=st.one_of(helpers.get_dtypes("integer")),
    k=helpers.ints(min_value=-10, max_value=10),
)
def test_triu_indices(
    *,
    n_rows,
    n_cols,
    input_dtype,
    k,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        device=on_device,
    )
