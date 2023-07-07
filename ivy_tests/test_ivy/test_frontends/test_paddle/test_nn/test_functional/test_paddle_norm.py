# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _batch_norm_helper(draw, *, min_dims=1):
    data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
    shape1, shape2, shape3, shape4 = draw(
        helpers.mutually_broadcastable_shapes(
            num_shapes=4, min_dims=min_dims, min_side=2
        )
    )
    shape = helpers.broadcast_shapes(shape1, shape2, shape3, shape4)
    if data_format == "NCHW":
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
    _, offset = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape3,
            min_value=-1001,
            max_value=999,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
        )
    )
    _, scale = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape4,
            min_value=-1001,
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
        offset[0],
        scale[0],
        eps,
        momentum,
        data_format,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.batch_norm",
    data=_batch_norm_helper(min_dims=2),
    training=st.booleans(),
)
def test_batch_norm(
    *,
    data,
    training,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    x_dtype, x, mean, variance, offset, scale, eps, momentum, data_format = data
    helpers.test_frontend_function(
        test_flags=test_flags,
        on_device=on_device,
        input_dtypes=x_dtype,
        frontend=frontend,
        fn_tree=fn_tree,
        x=x,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        epsilon=eps,
        training=training,
        momentum=momentum,
        data_format=data_format,
    )
