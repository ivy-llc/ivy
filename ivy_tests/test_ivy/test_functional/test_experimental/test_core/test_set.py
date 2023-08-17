from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.intersection",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shared_dtype=True,
        min_value=1,
        min_num_dims=1,
        min_dim_size=1,
        num_arrays=2,
    ),
    assume_unique=st.booleans(),
    return_indices=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_intersection(
    dtype_and_x,
    assume_unique,
    return_indices,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
        assume_unique=assume_unique,
        return_indices=return_indices,
    )
