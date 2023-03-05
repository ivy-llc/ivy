# global
from hypothesis import strategies as st

# local
import numpy as np
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
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
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
            helpers.list_of_size(
                x=st.sampled_from(interpolation_names),
                size=1,
            )
        )
        return dtype, values, axis, interpolation, q
    return dtype, values, axis


@handle_test(
    fn_tree="functional.ivy.experimental.median",
    dtype_x_axis=statistical_dtype_values(function="median"),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_median(
    *,
    dtype_x_axis,
    keep_dims,
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
        atol_=1e-02,
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
            available_dtypes=["int32", "int64"],
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
    *,
    dtype_x_shape,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
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
        on_device=on_device,
        indices=np.asarray(x[0], dtype=input_dtype[0]),
        shape=shape,
    )


# quantile
@handle_test(
    fn_tree="functional.ivy.experimental.quantile",
    dtype_and_x=statistical_dtype_values(function="quantile"),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_quantile(
    *,
    dtype_and_x,
    keep_dims,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
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


# bincount
@st.composite
def bincount_dtype_and_values(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32"],
            num_arrays=2,
            shared_dtype=True,
            min_num_dims=1,
            max_num_dims=1,
            min_dim_size=1,
            max_dim_size=10,
            min_value=0,
            max_value=10,
            allow_nan=False,
        )
    )
    dtype_and_x[1][1] = dtype_and_x[1][0]
    if draw(st.booleans()):
        dtype_and_x[1][1] = None

    min_length = draw(st.integers(min_value=0, max_value=10))
    return dtype_and_x, min_length


@handle_test(
    fn_tree="functional.ivy.experimental.bincount",
    dtype_and_x=bincount_dtype_and_values(),
    test_gradients=st.just(False),
)
def test_bincount(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype_and_x, min_length = dtype_and_x
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        weights=x[1],
        minlength=min_length,
    )
