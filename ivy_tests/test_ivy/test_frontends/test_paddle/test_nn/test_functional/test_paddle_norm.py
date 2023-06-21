# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
            max_value=999,
            min_value=-1001,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(shape[1],),
            max_value=999,
            min_value=0,
        )
    )
    _, others = draw(
        helpers.dtype_and_values(
            dtype=x_dtype * 3,
            shape=(shape[1],),
            max_value=999,
            min_value=-1001,
            num_arrays=3,
        )
    )
    return x_dtype, x[-1], others[0], others[1], others[2], variance[0]


@handle_frontend_test(
    fn_tree="paddle.nn.functional.batch_norm",
    data=_instance_and_batch_norm_helper(min_num_dims=2, min_dim_size=2),
    momentum=helpers.floats(min_value=0.01, max_value=0.1),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
    training=st.booleans(),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
    use_global_stats=st.booleans(),
    name=st.none(),
)
def test_paddle_batch_norm(
    *,
    data,
    momentum,
    eps,
    training,
    data_format,
    frontend,
    test_flags,
    fn_tree,
    use_global_stats,
    name=None,
):
    input_dtype, input, weight, bias, running_mean, running_var = data
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        atol=1e-1,
        rtol=1e-1,
        fn_tree=fn_tree,
        x=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
        data_format=data_format,
        use_global_stats=use_global_stats,
        name=name,
    )
