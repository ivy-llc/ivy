# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _generate_data_layer_norm(
    draw,
    *,
    available_dtypes,
    large_abs_safety_factor=100,
    small_abs_safety_factor=100,
    safety_factor_scale="log",
    min_num_dims=1,
    max_num_dims=5,
    valid_axis=True,
    allow_neg_axes=False,
    max_axes_size=1,
    force_int_axis=True,
    ret_shape=True,
    abs_smallest_val=1000,
    allow_inf=False,
    allow_nan=False,
    exclude_min=True,
    exclude_max=True,
    min_value=-1000,
    max_value=1000,
    shared_dtype=False,
    min_dim_size=1,
    max_dim_size=3,
    group=False,
):
    results = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            valid_axis=valid_axis,
            allow_neg_axes=allow_neg_axes,
            max_axes_size=max_axes_size,
            force_int_axis=force_int_axis,
            ret_shape=ret_shape,
        )
    )

    dtype, values, axis, shape = results

    if group:
        channel_size = shape[1]
        group_list = [*range(1, max_dim_size)]
        group_list = list(filter(lambda x: (channel_size % x == 0), group_list))
        group_size = draw(st.sampled_from(group_list))
        weight_shape = [shape[1]]
        bias_shape = [shape[1]]
    else:
        weight_shape = shape[axis:]
        bias_shape = shape[axis:]

    arg_dict = {
        "available_dtypes": dtype,
        "abs_smallest_val": abs_smallest_val,
        "min_value": min_value,
        "max_value": max_value,
        "large_abs_safety_factor": large_abs_safety_factor,
        "small_abs_safety_factor": small_abs_safety_factor,
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
    results_new_std = draw(helpers.dtype_and_values(shape=shape, **arg_dict))

    _, weight_values = results_weight
    _, bias_values = results_bias
    _, new_std_values = results_new_std

    axis = shape[axis:]
    if group:
        return dtype, values, weight_values, bias_values, group_size
    return dtype, values, axis, weight_values, bias_values, new_std_values


@st.composite
def _instance_and_batch_norm_helper(draw, *, min_num_dims=1, min_dim_size=1):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            min_num_dims=min_num_dims,
            max_num_dims=4,
            min_dim_size=min_dim_size,
            ret_shape=True,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(shape[1],),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
        )
    )
    _, mean = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(shape[1],),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
        )
    )
    _, others = draw(
        st.one_of(
            helpers.dtype_and_values(
                dtype=x_dtype * 2,
                shape=(shape[1],),
                large_abs_safety_factor=24,
                small_abs_safety_factor=24,
                safety_factor_scale="log",
                num_arrays=2,
            ),
            st.just(([None, None], [None, None])),
        )
    )
    momentum = draw(helpers.floats(min_value=0.01, max_value=0.1))
    eps = draw(helpers.floats(min_value=1e-5, max_value=0.1))
    return x_dtype, x[-1], others[0], others[1], mean[0], variance[0], momentum, eps


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="torch.nn.functional.batch_norm",
    data=_instance_and_batch_norm_helper(min_num_dims=2, min_dim_size=2),
    training=st.booleans(),
)
def test_torch_batch_norm(
    *,
    data,
    training,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtype, input, weight, bias, running_mean, running_var, momentum, eps = data
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        atol=1e-1,
        rtol=1e-1,
        fn_tree=fn_tree,
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


# group_norm
@handle_frontend_test(
    fn_tree="torch.nn.functional.group_norm",
    dtype_x_and_axis=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
        group=True,
    ),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
    test_with_out=st.just(False),
)
def test_torch_group_norm(
    dtype_x_and_axis,
    epsilon,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    dtype, x, weight, bias, group_size = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-1,
        rtol=1e-1,
        input=x[0],
        num_groups=group_size,
        weight=weight[0],
        bias=bias[0],
        eps=epsilon,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.instance_norm",
    data=_instance_and_batch_norm_helper(min_num_dims=3, min_dim_size=2),
    use_input_stats=st.booleans(),
)
def test_torch_instance_norm(
    *,
    data,
    use_input_stats,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtype, input, weight, bias, running_mean, running_var, momentum, eps = data
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-1,
        rtol=1e-1,
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.layer_norm",
    dtype_x_and_axis=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
)
def test_torch_layer_norm(
    *,
    dtype_x_and_axis,
    epsilon,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, axis, weight, bias, new_std = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        input=x[0],
        normalized_shape=axis,
        weight=weight[0],
        bias=bias[0],
        eps=epsilon,
    )
