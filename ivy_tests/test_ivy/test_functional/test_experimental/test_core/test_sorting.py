# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@st.composite
def _invert_permutation_helper(draw):
    return ["int64"], [
        np.array(
            draw(
                st.permutations(
                    list(range(draw(st.integers(min_value=3, max_value=10))))
                )
            )
        )
    ]


# invert_permutation
@handle_test(
    fn_tree="functional.ivy.experimental.invert_permutation",
    dtype_and_x=_invert_permutation_helper(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_invert_permutation(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# lexsort
@handle_test(
    fn_tree="functional.ivy.experimental.lexsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_lexsort(
    dtype_x_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend="numpy",
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        keys=x[0],
        axis=axis,
    )
