"""Collection of tests for unified neural network activations."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


# GELU
@handle_method(
    method_tree="stateful.activations.GELU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    approximate=st.booleans(),
    method_num_positional_args=helpers.num_positional_args(fn_name="GELU._forward"),
    test_gradients=st.just(True),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    test_gradients,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"approximate": approximate},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        atol_=1e-2,
        rtol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# GEGLU
@handle_method(
    method_tree="stateful.activations.GEGLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="GEGLU._forward"),
    test_gradients=st.just(True),
)
def test_geglu(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    # last dim must be even, this could replaced with a private helper
    assume(x[0].shape[-1] % 2 == 0)
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.ReLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="ReLU._forward"),
    test_gradients=st.just(True),
)
def test_relu(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.LeakyReLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(
            "float_and_complex", full=False, key="leaky_relu"
        ),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=-1e-4, max_value=1e-4),
    method_num_positional_args=helpers.num_positional_args(
        fn_name="LeakyReLU._forward"
    ),
    test_gradients=st.just(True),
)
def test_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"alpha": alpha},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.Softmax.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    method_num_positional_args=helpers.num_positional_args(fn_name="Softmax._forward"),
    test_gradients=st.just(True),
)
def test_softmax(
    *,
    dtype_and_x,
    axis,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "axis": axis},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.LogSoftmax.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    method_num_positional_args=helpers.num_positional_args(
        fn_name="LogSoftmax._forward"
    ),
    test_gradients=st.just(True),
)
def test_log_softmax(
    *,
    dtype_and_x,
    axis,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "axis": axis},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.Softplus.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    beta=st.one_of(helpers.number(min_value=0.1, max_value=10), st.none()),
    threshold=st.one_of(helpers.number(min_value=0.1, max_value=30), st.none()),
    method_num_positional_args=helpers.num_positional_args(fn_name="Softplus._forward"),
    test_gradients=st.just(True),
)
def test_softplus(
    *,
    dtype_and_x,
    beta,
    threshold,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "beta": beta, "threshold": threshold},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.Mish.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="Mish._forward"),
    test_gradients=st.just(True),
)
def test_mish(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="stateful.activations.SiLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="SiLU._forward"),
    test_gradients=st.just(True),
)
def test_silu(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# Sigmoid
@handle_method(
    method_tree="stateful.activations.Sigmoid.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="Sigmoid._forward"),
    test_gradients=st.just(True),
)
def test_sigmoid(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# Tanh
@handle_method(
    method_tree="stateful.activations.Tanh.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="Tanh._forward"),
    test_gradients=st.just(True),
)
def test_tanh(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# ReLU6
@handle_method(
    method_tree="stateful.activations.ReLU6.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="ReLU6._forward"),
    test_gradients=st.just(True),
)
def test_relu6(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# Hardswish
@handle_method(
    method_tree="stateful.activations.Hardswish.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(
        fn_name="Hardswish._forward"
    ),
    test_gradients=st.just(True),
)
def test_hardswish(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# Logit
@handle_method(
    method_tree="stateful.activations.Logit.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="Logit._forward"),
    eps=helpers.floats(min_value=1e-4, max_value=1e-2),
    test_gradients=st.just(True),
)
def test_logit(
    *,
    dtype_and_x,
    eps,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "eps": eps},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# PReLU
@handle_method(
    method_tree="stateful.activations.PReLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="PReLU._forward"),
    test_gradients=st.just(True),
)
def test_prelu(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "slope": x[1]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# SeLU
@handle_method(
    method_tree="stateful.activations.SeLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=2,
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="SeLU._forward"),
    test_gradients=st.just(True),
)
def test_selu(
    *,
    dtype_and_x,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )


# ELU
@handle_method(
    method_tree="stateful.activations.ELU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    method_num_positional_args=helpers.num_positional_args(fn_name="ELU._forward"),
    test_gradients=st.just(True),
    alpha=helpers.floats(min_value=0.1, max_value=1),
)
def test_elu(
    *,
    dtype_and_x,
    alpha,
    test_gradients,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=input_dtype,
        method_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_all_as_kwargs_np={"x": x[0], "alpha": alpha},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )
