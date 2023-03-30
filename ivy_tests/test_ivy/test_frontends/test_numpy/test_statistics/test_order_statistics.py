# global
from hypothesis import strategies as st
import numpy as np


# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

# Helpers #
# ------- #


@st.composite
def statistical_dtype_values(draw, *, function, min_value=None, max_value=None):
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
    
    if function == "percentile":
        q = draw(
            helpers.array_values(
                dtype=helpers.get_dtypes("float"),
                shape=helpers.get_shape(min_dim_size=1, max_num_dims=1, min_num_dims=1),
                min_value=0.0,
                max_value=100.0,
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


# percentile
@handle_frontend_test(
    fn_tree="numpy.percentile",
    dtype_x_q=statistical_dtype_values(function="percentile"),
    keep_dims=st.booleans(),
    overwrite_input=st.booleans(),
)
def test_numpy_percentile(
    dtype_x_q,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
    overwrite_input
):
    input_dtypes, x, axis, interpolation, q = dtype_x_q
    if isinstance(axis, tuple):
        axis = axis[0]

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        q=q,
        axis=axis,
        overwrite_input=overwrite_input,
        method=interpolation[0],
        out=None,
        keepdims=keep_dims,
    )
