# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


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
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape2,
            min_value=0,
            max_value=999,
        )
    )
    _, offset = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape3,
            min_value=-1001,
            max_value=999,
        )
    )
    _, scale = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape4,
            min_value=-1001,
            max_value=999,
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


@handle_test(
    fn_tree="functional.ivy.experimental.instance_norm",
    data=_instance_and_batch_norm_helper(min_dims=3),
    training=st.booleans(),
)
def test_instance_norm(
    *,
    data,
    training,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    test_flags.with_out = False
    x_dtype, x, scale, offset, mean, variance, eps, momentum, data_format = data
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
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


# batch_norm
@handle_test(
    fn_tree="functional.ivy.experimental.batch_norm",
    data=_instance_and_batch_norm_helper(min_dims=2, test_function="batch_norm"),
    training=st.booleans(),
)
def test_batch_norm(
    *,
    data,
    training,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    test_flags.with_out = False
    x_dtype, x, scale, offset, mean, variance, eps, momentum, data_format = data
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
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
