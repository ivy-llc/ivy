"""Collection of tests for unified neural network layers."""

# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@st.composite
def _generate_data_layer_norm(
    draw,
    *,
    available_dtypes,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
    min_num_dims=1,
    max_num_dims=5,
    valid_axis=True,
    allow_neg_axes=False,
    max_axes_size=1,
    force_int_axis=True,
    ret_shape=True,
    abs_smallest_val=0.1,
    allow_inf=False,
    allow_nan=False,
    exclude_min=False,
    exclude_max=False,
    min_value=-1e20,
    max_value=1e20,
    shared_dtype=False,
):
    results = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            min_value=min_value,
            max_value=max_value,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
            abs_smallest_val=abs_smallest_val,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            valid_axis=valid_axis,
            allow_neg_axes=allow_neg_axes,
            max_axes_size=max_axes_size,
            force_int_axis=force_int_axis,
            ret_shape=ret_shape,
        )
    )

    dtype, values, axis, shape = results

    weight_shape = shape[axis:]
    bias_shape = shape[axis:]

    arg_dict = {
        "available_dtypes": dtype,
        "abs_smallest_val": abs_smallest_val,
        "min_value": min_value,
        "max_value": max_value,
        "large_abs_safety_factor": large_abs_safety_factor,
        "small_abs_safety_factor": small_abs_safety_factor,
        "safety_factor_scale": safety_factor_scale,
        "allow_inf": allow_inf,
        "allow_nan": allow_nan,
        "exclude_min": exclude_min,
        "exclude_max": exclude_max,
        "min_num_dims": min_num_dims,
        "max_num_dims": max_num_dims,
        "shared_dtype": shared_dtype,
        "ret_shape": False,
    }

    results_weight = draw(helpers.dtype_and_values(shape=weight_shape, **arg_dict))
    results_bias = draw(helpers.dtype_and_values(shape=bias_shape, **arg_dict))

    _, weight_values = results_weight
    _, bias_values = results_bias

    return dtype, values, axis, weight_values, bias_values


@handle_test(
    fn_tree="functional.ivy.layer_norm",
    values_tuple=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    new_std=st.floats(min_value=0.01, max_value=0.1),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
)
def test_layer_norm(
    *,
    values_tuple,
    new_std,
    epsilon,
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
    dtype, x, normalize_axis, weight, bias = values_tuple
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=0.5,
        atol_=0.5,
        test_gradients=True,
        xs_grad_idxs=[[0, 0]],
        x=x[0],
        normalize_axis=normalize_axis,
        epsilon=epsilon,
        weight=weight[0],
        bias=bias[0],
        new_std=new_std,
    )
