# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.l2_normalize",
    dtype_and_x=helpers.arrays_and_axes(
        available_dtypes=helpers.get_dtypes("float"),
        num=1,
        return_dtype=True,
        force_int_axis=True,
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
    input_dtype, x, axis = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        x=x[0],
        axis=axis,
    )


# lp_normalize
@handle_test(
    fn_tree="functional.ivy.experimental.lp_normalize",
    dtype_and_x=helpers.arrays_and_axes(
        available_dtypes=helpers.get_dtypes("float"),
        num=1,
        return_dtype=True,
        force_int_axis=True,
    ),
    p=st.floats(min_value=0.1, max_value=2),
    test_gradients=st.just(False),
)
def test_lp_normalize(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
    p,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        axis=axis,
        p=p,
    )
