# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# difference
@handle_test(
    fn_tree="functional.ivy.experimental.difference",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["int32", "int64", "float32", "float64"],
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        max_num_dims=1,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    test_gradients=st.just(False),
)
def test_difference(
    dtype_and_x,
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
        ground_truth_backend="numpy",
        fw=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )
    