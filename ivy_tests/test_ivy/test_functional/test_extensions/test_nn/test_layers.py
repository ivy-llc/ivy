# global
from hypothesis import strategies as st

# local
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import ivy_tests.test_array_api.array_api_tests.hypothesis_helpers as hypothesis_helpers


# Helpers #
# ------- #


# vorbis_window
# TODO: Add container and array instance methods
@handle_test(
    fn_tree="functional.extensions.vorbis_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_vorbis_window(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=input_dtype,
    )


# flatten
@handle_test(
    fn_tree="functional.extensions.flatten",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(
            helpers.get_shape(min_num_dims=1, max_num_dims=5), key="flatten_shape"
        ),
        min_value=-100,
        max_value=100,
    ),
    axes=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(min_num_dims=1, max_num_dims=5), key="flatten_shape"
        ),
        allow_neg=True,
        sorted=True,
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
)
def test_flatten(
    dtype_and_x,
    axes,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    input_dtypes, x = dtype_and_x
    x = np.asarray(x[0], dtype=input_dtypes[0])

    if axes[1] == 0:
        start_dim, end_dim = axes[1], axes[0]
    elif axes[0] * axes[1] < 0:
        if x.ndim + min(axes) >= max(axes):
            start_dim, end_dim = max(axes), min(axes)
        else:
            start_dim, end_dim = min(axes), max(axes)
    else:
        start_dim, end_dim = axes[0], axes[1]
    helpers.test_function(
        input_dtypes=input_dtypes,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        x=x,
        start_dim=start_dim,
        end_dim=end_dim,
    )


# hann_window
# TODO: Add array instance methods
@handle_test(
    fn_tree="functional.extensions.hann_window",
    window_length=helpers.ints(min_value=1, max_value=10),
    input_dtype=helpers.get_dtypes("integer"),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float"),
)
def test_hann_window(
    window_length,
    input_dtype,
    periodic,
    dtype,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        window_length=window_length,
        periodic=periodic,
        dtype=dtype[0],
    )


@handle_test(
    fn_tree="functional.extensions.max_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
)
def test_max_pool2d(
    *,
    x_k_s_p,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.extensions.max_pool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
)
def test_max_pool1d(
    *,
    x_k_s_p,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


# kaiser_window
# TODO: Add array instance methods
@handle_test(
    fn_tree="functional.extensions.kaiser_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=(1, 1),
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=0, max_value=5),
    dtype=helpers.get_dtypes("float"),
)
def test_kaiser_window(
    dtype_and_x,
    periodic,
    beta,
    dtype,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        window_length=x[0],
        periodic=periodic,
        beta=beta,
        dtype=dtype[0],
    )


def _st_tuples_or_int(n_pairs):
    return st.one_of(
        hypothesis_helpers.tuples(
            st.tuples(
                st.integers(min_value=1, max_value=4),
                st.integers(min_value=1, max_value=4),
            ),
            min_size=n_pairs,
            max_size=n_pairs,
        ),
        helpers.ints(min_value=1, max_value=4),
    )


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            ret_shape=True,
            min_num_dims=1,
        )
    )
    ndim = len(shape)
    pad_width = draw(_st_tuples_or_int(ndim))
    stat_length = draw(_st_tuples_or_int(ndim))
    constant_values = draw(_st_tuples_or_int(ndim))
    end_values = draw(_st_tuples_or_int(ndim))
    return dtype, value, pad_width, stat_length, constant_values, end_values


@handle_test(
    fn_tree="functional.extensions.pad",
    dtype_and_input_and_other=_pad_helper(),
    mode=st.sampled_from(
        [
            "constant",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
        ]
    ),
    reflect_type=st.sampled_from(["even", "odd"]),
)
def test_pad(
    *,
    dtype_and_input_and_other,
    mode,
    reflect_type,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    (
        dtype,
        value,
        pad_width,
        stat_length,
        constant_values,
        end_values,
    ) = dtype_and_input_and_other
    helpers.test_function(
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        ground_truth_backend="numpy",
        input=value[0],
        pad_width=pad_width,
        mode=mode,
        stat_length=stat_length,
        constant_values=constant_values,
        end_values=end_values,
        reflect_type=reflect_type,
        out=None,
    )


# kaiser_bessel_derived_window
# TODO: Add array instance methods
@handle_test(
    fn_tree="functional.extensions.kaiser_bessel_derived_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(1, 1),
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
)
def test_kaiser_bessel_derived_window(
    dtype_and_x,
    periodic,
    beta,
    dtype,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        window_length=x[0],
        periodic=periodic,
        beta=beta,
        dtype=dtype[0],
    )


# hamming_window
# TODO: Add array instance methods
@handle_test(
    fn_tree="functional.extensions.hamming_window",
    window_length=helpers.ints(min_value=1, max_value=10),
    input_dtype=helpers.get_dtypes("integer"),
    periodic=st.booleans(),
    alpha=st.floats(min_value=1, max_value=5),
    beta=st.floats(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
)
def test_hamming_window(
    window_length,
    input_dtype,
    periodic,
    alpha,
    beta,
    dtype,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        test_gradients=test_gradients,
        fw=backend_fw,
        fn_name=fn_name,
        window_length=window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        dtype=dtype[0],
    )
