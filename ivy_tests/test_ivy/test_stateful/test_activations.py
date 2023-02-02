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
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
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
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    # last dim must be even, this could replaced with a private helper
    assume(x[0].shape[-1] % 2 == 0)
    helpers.test_method(
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
    )
