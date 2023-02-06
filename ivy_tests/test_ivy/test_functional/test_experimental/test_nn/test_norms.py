# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers
@st.composite
def _generate_data_instance_norm(
    draw,
    *,
    available_dtypes,
    large_abs_safety_factor=8,
    small_abs_safety_factor=8,
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
    x_shape = draw(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        )
    )
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
            shape=x_shape,
            valid_axis=valid_axis,
            allow_neg_axes=allow_neg_axes,
            max_axes_size=max_axes_size,
            force_int_axis=force_int_axis,
            ret_shape=ret_shape,
        )
    )
    dtype, values, axis, shape = results
    data_format = draw(st.sampled_from(["NHWC", "NCHW"]))
    if data_format == "NHWC":
        weight_shape = x_shape[3]
        bias_shape = x_shape[3]
    else:
        weight_shape = x_shape[1]
        bias_shape = x_shape[1]
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

    results_weight = draw(helpers.dtype_and_values(shape=[weight_shape], **arg_dict))
    results_bias = draw(helpers.dtype_and_values(shape=[bias_shape], **arg_dict))

    _, weight_values = results_weight
    _, bias_values = results_bias

    return dtype, values, axis, weight_values, bias_values, data_format


# instance_norm
@handle_test(
    fn_tree="functional.ivy.experimental.instance_norm",
    values_tuple=_generate_data_instance_norm(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_num_dims=4,
    ),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
    momentum=st.floats(min_value=0.01, max_value=0.1),
    affine=st.booleans(),
    track_running_stats=st.booleans(),
)
def test_instance_norm(
    *,
    values_tuple,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
    momentum,
    affine,
    track_running_stats,
):
    dtype, x, normalize_axis, scale, b, data_format = values_tuple
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=0.5,
        atol_=0.5,
        xs_grad_idxs=[[0, 0]],
        x=x[0],
        scale=scale[0],
        bias=b[0],
        eps=epsilon,
        momentum=momentum,
        data_format=data_format,
        running_mean=scale[0],
        running_stddev=b[0],
        affine=affine,
        track_running_stats=track_running_stats,
    )

@handle_test(
    fn_tree="functional.ivy.experimental.l2_normalize",
    dtype_and_x=helpers.arrays_and_axes(
        available_dtypes=helpers.get_dtypes("float"),
        num=1,
        returndtype=True,
        force_int_axis=True,
    ),
    test_gradients=st.just(False),
)
def test_l2_normalize(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        axis=axis,
    )

@handle_test(
    fn_tree="functional.ivy.experimental.lp_normalize",
    dtype_and_x=helpers.arrays_and_axes(
        available_dtypes=helpers.get_dtypes("float"),
        num=1,
        returndtype=True,
        force_int_axis=True,
    ),
    test_gradients=st.just(False),
)
def test_lp_normalize(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
    
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        axis=axis,
        p=2,
    )
