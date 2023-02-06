# global
# local
import numpy as np
from hypothesis import strategies as st, settings

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers #
# ------- #

@st.composite
def statistical_dtype_values(draw, *, function):
    large_abs_safety_factor = 2
    small_abs_safety_factor = 2
    if function in ["mean", "median", "std", "var"]:
        large_abs_safety_factor = 24
        small_abs_safety_factor = 24
    n = 1
    min_value = None
    max_value = None
    force_int_axis = False
    shape = None
    shared_dtype = False
    if function == "histogram":
        n = 2
        min_value = -20
        max_value = 20
        force_int_axis = True
        shape = draw(helpers.get_shape(min_num_dims=1))
        shared_dtype = True
    available_dtypes = draw(helpers.get_dtypes("float"))
    if "bfloat16" in available_dtypes:
        available_dtypes.remove("bfloat16")
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=5,
            min_value=min_value,
            max_value=max_value,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
            num_arrays=n,
            force_int_axis=force_int_axis,
            shape=shape,
            shared_dtype=shared_dtype,
        )
    )
    shape = values[0].shape
    size = values[0].size
    max_correction = np.min(shape)
    if function == "var" or function == "std":
        if size == 1:
            correction = 0
        elif isinstance(axis, int):
            correction = draw(
                helpers.ints(min_value=0, max_value=shape[axis] - 1)
                | helpers.floats(min_value=0, max_value=shape[axis] - 1)
            )
            return dtype, values, axis, correction
        else:
            correction = draw(
                helpers.ints(min_value=0, max_value=max_correction - 1)
                | helpers.floats(min_value=0, max_value=max_correction - 1)
            )
        return dtype, values, axis, correction

    if function == "quantile":
        q = draw(
            helpers.array_values(
                dtype=helpers.get_dtypes("float"),
                shape=helpers.get_shape(min_dim_size=1, max_num_dims=1, min_num_dims=1),
                min_value=0.0,
                max_value=1.0,
                exclude_max=False,
                exclude_min=False,
            )
        )

        interpolation_names = ["linear", "lower", "higher", "midpoint", "nearest"]
        interpolation = draw(
            helpers.lists(
                arg=st.sampled_from(interpolation_names), min_size=1, max_size=1
            )
        )
        return dtype, values, axis, interpolation, q

    if function == "histogram":
        dtype, values, dtype_out = draw(
            helpers.get_castable_dtype(
                available_dtypes, dtype[0], values
            )
        )
        bins = draw(
            helpers.array_values(
                min_value=1,
                max_value=100,
                dtype=dtype,
                large_abs_safety_factor=large_abs_safety_factor,
                small_abs_safety_factor=small_abs_safety_factor,
                safety_factor_scale="log",
                shape=draw(
                    helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=2,
                                      max_dim_size=10)
                ),
            )
        )
        bins = sorted(set(bins))
        if len(bins) == 1:
            bins = int(bins[0])
            range = (-10, 10)
        else:
            range = None
        return dtype, values, axis, dtype_out, bins, range
    return dtype, values, axis


@handle_test(
    fn_tree="functional.ivy.experimental.histogram",
    statistical_dtype_values=statistical_dtype_values(function="histogram"),
    extend_lower_interval=st.booleans(),
    extend_upper_interval=st.booleans(),
    density=st.booleans(),
    test_gradients=st.just(False),
)
def test_histogram(
    *,
    statistical_dtype_values,
    extend_lower_interval,
    extend_upper_interval,
    density,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    input_dtype, values, axis, dtype_out, bins, range = statistical_dtype_values
    helpers.test_function(
        a=values[0],
        bins=bins,
        axis=axis,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype_out,
        range=range,
        weights=values[1],
        density=density,
        input_dtypes=[input_dtype],
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        ground_truth_backend=ground_truth_backend,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.median",
    dtype_x_axis=statistical_dtype_values(function="median"),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
)
def test_median(
    *,
    dtype_x_axis,
    keep_dims,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        input=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# nanmean
@handle_test(
    fn_tree="functional.ivy.experimental.nanmean",
    dtype_x_axis=statistical_dtype_values(function="nanmean"),
    keep_dims=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
)
def test_nanmean(
    *,
    dtype_x_axis,
    keep_dims,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype[0],
    )


# unravel_index
@st.composite
def max_value_as_shape_prod(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    dtype_and_x = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=np.prod(shape) - 1,
        )
    )
    return dtype_and_x, shape


@handle_test(
    fn_tree="functional.ivy.experimental.unravel_index",
    dtype_x_shape=max_value_as_shape_prod(),
    test_gradients=st.just(False),
)
def test_unravel_index(
    dtype_x_shape,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype_and_x, shape = dtype_x_shape
    input_dtype, x = dtype_and_x[0], dtype_and_x[1]
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        indices=np.asarray(x[0], dtype=input_dtype[0]),
        shape=shape,
    )


# quantile
@handle_test(
    fn_tree="functional.ivy.experimental.quantile",
    dtype_and_x=statistical_dtype_values(function="quantile"),
    keep_dims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="quantile"),
    test_gradients=st.just(False),
)
def test_quantile(
    *,
    dtype_and_x,
    keep_dims,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x[0],
        q=q,
        axis=axis,
        interpolation=interpolation[0],
        keepdims=keep_dims,
    )


# corrcoef
@handle_test(
    fn_tree="functional.ivy.experimental.corrcoef",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        abs_smallest_val=1e-5,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    rowvar=st.booleans(),
    test_gradients=st.just(False),
)
def test_corrcoef(
    *,
    dtype_and_x,
    rowvar,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        y=x[1],
        rowvar=rowvar,
    )
