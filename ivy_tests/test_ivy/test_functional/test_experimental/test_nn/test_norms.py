# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@st.composite
def _instance_and_batch_norm_helper(draw, *, min_num_dims=1, min_dim_size=1):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            min_num_dims=min_num_dims,
            max_num_dims=5,
            min_dim_size=min_dim_size,
            ret_shape=True,
            max_value=999,
            min_value=-1001,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(shape[-1],),
            max_value=999,
            min_value=0,
        )
    )
    _, others = draw(
        helpers.dtype_and_values(
            dtype=x_dtype * 3,
            shape=(shape[-1],),
            max_value=999,
            min_value=-1001,
            num_arrays=3,
        )
    )
    return x_dtype, x[-1], others[0], others[1], others[2], variance[0]


@handle_test(
    fn_tree="functional.ivy.experimental.instance_norm",
    data=_instance_and_batch_norm_helper(min_num_dims=3, min_dim_size=2),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
    momentum=helpers.floats(min_value=0.0, max_value=1.0),
    training=st.booleans(),
)
def test_instance_norm(
    *,
    data,
    eps,
    momentum,
    training,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    x_dtype, x, scale, offset, mean, variance = data
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
    )


# batch_norm
@handle_test(
    fn_tree="functional.ivy.experimental.batch_norm",
    data=_instance_and_batch_norm_helper(min_num_dims=2, min_dim_size=2),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
    momentum=helpers.floats(min_value=0.0, max_value=1.0),
    training=st.booleans(),
)
def test_batch_norm(
    *,
    data,
    eps,
    momentum,
    training,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    x_dtype, x, scale, offset, mean, variance = data
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
    )
