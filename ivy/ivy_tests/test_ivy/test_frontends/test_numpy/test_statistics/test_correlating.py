# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# correlate
@handle_frontend_test(
    fn_tree="numpy.correlate",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    mode=st.sampled_from(["valid", "same", "full"]),
)
def test_numpy_correlate(
    dtype_and_x,
    mode,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        mode=mode,
    )
