# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import ivy


# --- Helpers --- #
# --------------- #


@st.composite
def _generate_data_layer_norm(
    draw,
    *,
    available_dtypes,
    large_abs_safety_factor=20,
    small_abs_safety_factor=20,
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
    normalized_idxs = list(range(axis, len(shape)))

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

    return dtype, values, normalized_idxs, weight_values, bias_values


@st.composite
def _group_norm_helper(draw):
    data_format = draw(st.sampled_from(["NSC", "NCS"]))
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=4, min_dim_size=2, max_dim_size=4
        )
    )
    channel_size = shape[-1]
    group_list = [*range(1, 4)]
    group_list = list(filter(lambda x: (channel_size % x == 0), group_list))
    num_groups = draw(st.sampled_from(group_list))
    if data_format == "NCS":
        shape = (shape[0], shape[-1], *shape[1:-1])
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float",
            ),
            shape=shape,
            large_abs_safety_factor=50,
            small_abs_safety_factor=50,
            safety_factor_scale="log",
        )
    )
    _, offset = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(channel_size,),
            large_abs_safety_factor=50,
            small_abs_safety_factor=50,
            safety_factor_scale="log",
        )
    )

    _, scale = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(channel_size,),
            large_abs_safety_factor=50,
            small_abs_safety_factor=50,
            safety_factor_scale="log",
        )
    )
    eps = draw(helpers.floats(min_value=1e-5, max_value=0.1))
    return x_dtype, x[0], num_groups, data_format, scale[0], offset[0], eps


@st.composite
def _instance_and_batch_norm_helper(draw, *, min_dims=1, test_function="instance_norm"):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"
    data_format = draw(st.sampled_from(["NSC", "NCS"]))
    shape1, shape2, shape3, shape4 = draw(
        helpers.mutually_broadcastable_shapes(
            num_shapes=4, min_dims=min_dims, min_side=2
        )
    )
    shape = helpers.broadcast_shapes(shape1, shape2, shape3, shape4)

    if (test_function == "instance_norm") or (is_torch_backend and not mixed_fn_compos):
        shape1 = shape2 = shape3 = shape4 = (shape[-1],)

    if data_format == "NCS":
        shape = (shape[0], shape[-1], *shape[1:-1])

    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float",
                mixed_fn_compos=mixed_fn_compos,
            ),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            max_value=999,
            min_value=-1001,
        )
    )

    _, mean = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape1,
            min_value=-1001,
            max_value=999,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape2,
            min_value=0,
            max_value=999,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
        )
    )
    _, offset = draw(
        st.one_of(
            helpers.dtype_and_values(
                dtype=x_dtype,
                shape=shape3,
                min_value=-1001,
                max_value=999,
                large_abs_safety_factor=24,
                small_abs_safety_factor=24,
                safety_factor_scale="log",
            ),
            st.just(([None], [None])),
        )
    )
    _, scale = draw(
        st.one_of(
            helpers.dtype_and_values(
                dtype=x_dtype,
                shape=shape4,
                min_value=-1001,
                max_value=999,
                large_abs_safety_factor=24,
                small_abs_safety_factor=24,
                safety_factor_scale="log",
            ),
            st.just(([None], [None])),
        )
    )
    eps = draw(
        helpers.floats(min_value=1e-5, max_value=0.1, mixed_fn_compos=mixed_fn_compos)
    )
    momentum = draw(
        helpers.floats(min_value=0.0, max_value=1.0, mixed_fn_compos=mixed_fn_compos)
    )
    return (
        x_dtype,
        x[0],
        mean[0],
        variance[0],
        offset[0],
        scale[0],
        eps,
        momentum,
        data_format,
    )


# --- Main --- #
# ------------ #


# batch_norm
@handle_test(
    fn_tree="functional.ivy.experimental.batch_norm",
    data=_instance_and_batch_norm_helper(min_dims=2, test_function="batch_norm"),
    training=st.booleans(),
)
def test_batch_norm(*, data, training, test_flags, backend_fw, fn_name, on_device):
    x_dtype, x, mean, variance, offset, scale, eps, momentum, data_format = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        xs_grad_idxs=[[0, 0]],
        rtol_=1e-2,
        atol_=1e-2,
        input_dtypes=x_dtype,
        x=x,
        mean=mean,
        variance=variance,
        scale=scale,
        offset=offset,
        eps=eps,
        training=training,
        momentum=momentum,
        data_format=data_format,
    )


# group_norm
@handle_test(
    fn_tree="functional.ivy.experimental.group_norm",
    data=_group_norm_helper(),
)
def test_group_norm(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    x_dtype, x, num_groups, data_format, scale, offset, eps = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        xs_grad_idxs=[[0, 0]],
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=x_dtype,
        x=x,
        num_groups=num_groups,
        scale=scale,
        offset=offset,
        eps=eps,
        data_format=data_format,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.instance_norm",
    data=_instance_and_batch_norm_helper(min_dims=3),
    training=st.booleans(),
)
def test_instance_norm(*, data, training, test_flags, backend_fw, fn_name, on_device):
    x_dtype, x, mean, variance, offset, scale, eps, momentum, data_format = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        xs_grad_idxs=[[0, 0]],
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=x_dtype,
        x=x,
        mean=mean,
        variance=variance,
        scale=scale,
        offset=offset,
        eps=eps,
        training=training,
        momentum=momentum,
        data_format=data_format,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.l1_normalize",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"), valid_axis=True
    ),
)
def test_l1_normalize(*, dtype_values_axis, test_flags, backend_fw, fn_name, on_device):
    x_dtype, x, axis = dtype_values_axis
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=x_dtype,
        x=x,
        axis=axis,
    )


# layer_norm
@handle_test(
    fn_tree="functional.ivy.experimental.layer_norm",
    values_tuple=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    new_std=st.floats(min_value=0.01, max_value=0.1),
    eps=st.floats(min_value=0.01, max_value=0.1),
)
def test_layer_norm(
    *, values_tuple, new_std, eps, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, normalized_idxs, scale, offset = values_tuple
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=0.5,
        atol_=0.5,
        xs_grad_idxs=[[0, 0]],
        x=x[0],
        normalized_idxs=normalized_idxs,
        eps=eps,
        scale=scale[0],
        offset=offset[0],
        new_std=new_std,
    )
