# global
from hypothesis import strategies as st
import hypothesis.extra.numpy as nph

# local
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.l2_normalize",
    dtype_and_x=helpers.arrays_and_axes(
        num=1,
    ),
    test_gradients=st.just(False),
)
def test_l2_normalize(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    x, axis = dtype_and_x

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=[x[0].dtype],
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        axis=axis
    )

