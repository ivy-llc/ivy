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


@st.composite
def _histogram_helper(draw):
    dtype_input = draw(st.sampled_from(draw(helpers.get_dtypes("float"))))
    bins = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=(draw(helpers.ints(min_value=1, max_value=10)),),
            abs_smallest_val=-10,
            min_value=-10,
            max_value=10,
        )
    )
    bins = np.asarray(sorted(set(bins)), dtype=dtype_input)
    if len(bins) == 1:
        bins = int(abs(bins[0]))
        if bins == 0:
            bins = 1
        if dtype_input in draw(helpers.get_dtypes("unsigned")):
            range = (
                draw(
                    helpers.floats(
                        min_value=0, max_value=10, exclude_min=False, exclude_max=False
                    )
                ),
                draw(
                    helpers.floats(
                        min_value=11, max_value=20, exclude_min=False, exclude_max=False
                    )
                ),
            )
        else:
            range = (
                draw(helpers.floats(min_value=-10, max_value=0)),
                draw(helpers.floats(min_value=1, max_value=10)),
            )
        range = draw(st.sampled_from([range, None]))
    else:
        range = None
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=5
        )
    )
    a = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    weights = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    # weights = draw(st.sampled_from([weights, None]))
    axes = draw(
        helpers.get_axis(
            shape=shape,
            # TODO: negative axes
            allow_neg=False,
            min_size=1,
            max_size=10,
        )
    )
    dtype_out = draw(
        st.sampled_from(
            draw(
                helpers.get_castable_dtype(
                    draw(helpers.get_dtypes("float")), str(dtype_input)
                )
            )
        )
    )
    if range:
        if np.min(a) < range[0]:
            extend_lower_interval = True
        else:
            extend_lower_interval = draw(st.booleans())
        if np.max(a) > range[1]:
            extend_upper_interval = True
        else:
            extend_upper_interval = draw(st.booleans())
    else:
        if isinstance(bins, int):
            extend_lower_interval = draw(st.booleans())
            extend_upper_interval = draw(st.booleans())
        else:
            if np.min(a) < bins[0]:
                extend_lower_interval = True
            else:
                extend_lower_interval = draw(st.booleans())
            if np.max(a) > bins[-1]:
                extend_upper_interval = True
            else:
                extend_upper_interval = draw(st.booleans())
    density = draw(st.booleans())
    return (
        a,
        bins,
        axes,
        extend_lower_interval,
        extend_upper_interval,
        dtype_out,
        range,
        weights,
        density,
        dtype_input,
    )


# TODO: - Issue: https://github.com/tensorflow/probability/issues/1712
#       - Fix: https://github.com/tensorflow/probability/commit/bcca631f01c855425710fe3
#       b8947192b71e310dd
#       - Error message from Tensorflow: 'Number of dimensions of `x` and `weights`
#       must coincide. Found: x has <nd1>, weights has <nd2>'
#       - Error description: typo that throws unintended exceptions when using both
#       weights and multiple axis.
#       - This test is going to be fixed in the next tensorflow_probability release by
#       the commit when it merges.
@handle_test(
    fn_tree="functional.ivy.experimental.histogram",
    values=_histogram_helper(),
    test_gradients=st.just(False),
)
def test_histogram(
    *,
    values,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
    on_device,
):
    (
        a,
        bins,
        axis,
        extend_lower_interval,
        extend_upper_interval,
        dtype,
        range,
        weights,
        density,
        dtype_input,
    ) = values
    helpers.test_function(
        a=a,
        bins=bins,
        axis=axis,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        range=range,
        weights=weights,
        density=density,
        input_dtypes=[dtype_input],
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        ground_truth_backend=ground_truth_backend,
        on_device=on_device,
    )


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
