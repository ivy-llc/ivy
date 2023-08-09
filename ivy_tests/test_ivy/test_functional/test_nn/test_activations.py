"""Collection of tests for unified neural network activation functions."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# relu
@handle_test(
    fn_tree="functional.ivy.relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_relu(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# leaky_relu
@handle_test(
    fn_tree="functional.ivy.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(
            "float_and_complex", full=False, key="leaky_relu"
        ),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=-1e-4, max_value=1e-4),
)
def test_leaky_relu(*, dtype_and_x, alpha, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        alpha=alpha,
    )


# gelu
@handle_test(
    fn_tree="functional.ivy.gelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=1,
        small_abs_safety_factor=1,
        safety_factor_scale="linear",
        min_value=-1e4,
        max_value=1e4,
    ),
    approximate=st.booleans(),
)
def test_gelu(*, dtype_and_x, approximate, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    if "complex" in str(x[0].dtype):
        approximate = True
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-2,
        rtol_=1e-2,
        x=x[0],
        approximate=approximate,
    )


# sigmoid
@handle_test(
    fn_tree="functional.ivy.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_sigmoid(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# softmax
@handle_test(
    fn_tree="functional.ivy.softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=st.one_of(
        helpers.ints(min_value=-1, max_value=0),
        st.none(),
    ),
)
def test_softmax(*, dtype_and_x, axis, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        axis=axis,
    )


# softplus
@handle_test(
    fn_tree="functional.ivy.softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    beta=st.one_of(helpers.number(min_value=0.1, max_value=10), st.none()),
    threshold=st.one_of(helpers.number(min_value=0.1, max_value=30), st.none()),
)
def test_softplus(
    *, dtype_and_x, beta, threshold, test_flags, backend_fw, fn_name, on_device
):
    assume(beta != 0)
    assume(threshold != 0)
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        beta=beta,
        threshold=threshold,
    )


# log_softmax
@handle_test(
    fn_tree="functional.ivy.log_softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=st.one_of(helpers.ints(min_value=-1, max_value=0), st.none()),
)
def test_log_softmax(*, dtype_and_x, axis, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        axis=axis,
    )


# mish
@handle_test(
    fn_tree="functional.ivy.mish",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_mish(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
    )


# hardswish
@handle_test(
    fn_tree="functional.ivy.hardswish",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_hardswish(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )
