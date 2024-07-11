# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# celu
@handle_test(
    fn_tree="functional.ivy.experimental.celu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=0.1, max_value=1.0),
    complex_mode=st.sampled_from(["jax", "split", "magnitude"]),
)
def test_celu(
    *, dtype_and_x, alpha, complex_mode, test_flags, backend_fw, fn_name, on_device
):
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
        complex_mode=complex_mode,
    )


# elu
@handle_test(
    fn_tree="functional.ivy.experimental.elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    alpha=st.one_of(
        st.floats(min_value=0.10, max_value=1.0),
    ),
)
def test_elu(
    *,
    dtype_and_x,
    alpha,
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
        alpha=alpha,
    )


# hardshrink
@handle_test(
    fn_tree="functional.ivy.experimental.hardshrink",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    threshold=st.one_of(
        st.floats(min_value=0.0, max_value=1e30),
    ),
)
def test_hardshrink(
    *, dtype_and_x, threshold, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        lambd=threshold,
    )


# hardsilu
@handle_test(
    fn_tree="functional.ivy.experimental.hardsilu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_hardsilu(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# hardtanh
@handle_test(
    fn_tree="functional.ivy.experimental.hardtanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    min_val=st.one_of(
        st.floats(min_value=-10.0, max_value=-1.0),
    ),
    max_val=st.one_of(
        st.floats(min_value=1.0, max_value=10.0),
    ),
)
def test_hardtanh(
    *,
    dtype_and_x,
    min_val,
    max_val,
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
        min_val=min_val,
        max_val=max_val,
    )


# logit
@handle_test(
    fn_tree="functional.ivy.experimental.logit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_logit(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# logsigmoid
@handle_test(
    fn_tree="functional.ivy.experimental.logsigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        safety_factor_scale="log",
        large_abs_safety_factor=120,
    ),
    test_with_out=st.just(False),
)
def test_logsigmoid(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    test_flags.num_positional_args = len(x)
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=x[0],
    )


# prelu
@handle_test(
    fn_tree="prelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="prelu"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    slope=helpers.array_values(
        dtype=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(), key="prelu"),
    ),
)
def test_prelu(*, dtype_and_x, slope, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        slope=slope,
    )


# relu6
@handle_test(
    fn_tree="functional.ivy.experimental.relu6",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        min_value=1e-15,
    ),
    complex_mode=st.sampled_from(["jax", "split", "magnitude"]),
)
def test_relu6(
    *, dtype_and_x, complex_mode, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        complex_mode=complex_mode,
    )


# scaled_tanh
@handle_test(
    fn_tree="functional.ivy.experimental.scaled_tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_dim_size=1,
        min_num_dims=1,
    ),
    alpha=st.floats(min_value=0.1, max_value=5.0),
    beta=st.floats(min_value=0.1, max_value=5.0),
    ground_truth_backend="paddle",
)
def test_scaled_tanh(
    *, dtype_and_x, alpha, beta, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-5,
        atol_=1e-5,
        x=x[0],
        alpha=alpha,
        beta=beta,
    )


# selu
@handle_test(
    fn_tree="functional.ivy.experimental.selu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        safety_factor_scale="log",
        small_abs_safety_factor=20,
    ),
    test_with_out=st.just(False),
)
def test_selu(*, dtype_and_input, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input = dtype_and_input
    test_flags.num_positional_args = len(input)
    helpers.test_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-2,
        x=input[0],
    )


# silu
@handle_test(
    fn_tree="functional.ivy.experimental.silu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_silu(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
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


# softshrink
@handle_test(
    fn_tree="functional.ivy.experimental.softshrink",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    threshold=st.one_of(
        st.floats(min_value=0.0, max_value=1e30),
    ),
)
def test_softshrink(
    *, dtype_and_x, threshold, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        lambd=threshold,
    )


# tanhshrink
@handle_test(
    fn_tree="functional.ivy.experimental.tanhshrink",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_tanhshrink(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
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


# threshold
@handle_test(
    fn_tree="functional.ivy.experimental.threshold",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    threshold=st.one_of(
        st.floats(min_value=-1e30, max_value=1e30),
    ),
    value=st.one_of(
        st.floats(min_value=-1e30, max_value=1e30),
    ),
)
def test_threshold(
    *, dtype_and_x, threshold, value, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        threshold=threshold,
        value=value,
    )


# thresholded_relu
@handle_test(
    fn_tree="functional.ivy.experimental.thresholded_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    threshold=st.one_of(
        st.floats(min_value=-0.10, max_value=10.0),
    ),
)
def test_thresholded_relu(
    *, dtype_and_x, threshold, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        threshold=threshold,
    )
