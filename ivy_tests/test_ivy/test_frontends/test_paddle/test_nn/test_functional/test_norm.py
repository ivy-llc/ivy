# global
from hypothesis import strategies as st
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# instance_norm
@st.composite
def _instance_and_batch_norm_helper(draw, *, min_dims=1, test_function="instance_norm"):
    data_format = draw(st.sampled_from(["NSC", "NCS"]))
    shape1, shape2, shape3, shape4 = draw(
        helpers.mutually_broadcastable_shapes(
            num_shapes=4, min_dims=min_dims, min_side=2
        )
    )
    shape = helpers.broadcast_shapes(shape1, shape2, shape3, shape4)
    if test_function == "instance_norm":
        shape1 = shape2 = shape3 = shape4 = (shape[-1],)

    if data_format == "NCS":
        shape = (shape[0], shape[-1], *shape[1:-1])

    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float",
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

    _, weights = draw(
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

    _, bias = draw(
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

    eps = draw(helpers.floats(min_value=1e-5, max_value=0.1))
    momentum = draw(helpers.floats(min_value=0.0, max_value=1.0))
    return (
        x_dtype,
        x[0],
        mean[0],
        variance[0],
        weights[0],
        bias[0],
        eps,
        momentum,
        data_format,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.instance_norm",
    data=_instance_and_batch_norm_helper(min_dims=3),
    use_input_stats=st.booleans(),
)
def test_paddle_instance_norm(
        *,
        data,
        use_input_stats,
        frontend,
        test_flags,
        fn_tree,
):
    x_dtype, x, running_mean, running_var, weights, bias, eps, momentum, data_format = data
    helpers.test_frontend_function(
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        input_dtypes=x_dtype,
        x=x,
        running_mean=running_mean,
        running_var=running_var,
        weight=weights,
        bias=bias,
        eps=eps,
        momentum=momentum,
        data_format=data_format,
        use_input_stats=use_input_stats,
    )
