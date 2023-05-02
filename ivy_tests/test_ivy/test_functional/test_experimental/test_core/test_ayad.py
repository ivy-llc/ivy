# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# in_top_k
@handle_frontend_test(
    fn_tree="tensorflow.math.in_top_k",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    k=st.integers(min_value=0, max_value=5),
    test_with_out=st.just(False),
)
def test_tensorflow_in_top_k(
    *, dtype_and_x, frontend, test_flags, fn_tree, on_device, k
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        target=x[0],
        pred=x[1],
        k=k,
    )
