# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_helpers


# corrcoef
@handle_frontend_test(
    fn_tree="numpy.corrcoef",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        abs_smallest_val=1e-5,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-100,
        max_value=100,
    ),
    rowvar=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_numpy_corrcoef(
    dtype_and_x,
    rowvar,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    dtype,
    backend_fw,
):
    input_dtypes, x = dtype_and_x
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        rowvar=rowvar,
        dtype=dtype[0],
        backend_to_test=backend_fw,
    )


# correlate
@handle_frontend_test(
    fn_tree="numpy.correlate",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        shared_dtype=True,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    mode=st.sampled_from(["valid", "same", "full"]),
    test_with_out=st.just(False),
)
def test_numpy_correlate(
    dtype_and_x,
    mode,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        a=xs[0],
        v=xs[1],
        mode=mode,
    )
