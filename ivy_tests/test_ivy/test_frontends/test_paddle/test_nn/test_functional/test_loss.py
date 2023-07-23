# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.nn.functional.binary_cross_entropy_with_logits",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=1,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_binary_cross_entropy_with_logits(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    # TODO: paddle's implementation of pos_weight is wrong
    # https://github.com/PaddlePaddle/Paddle/pull/54869
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            x_dtype[1],
            weight_dtype[0],
        ],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logit=x[0],
        label=x[1],
        weight=weight[0],
        reduction=reduction,
    )


# mse_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.mse_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_mse_loss(
    dtype_and_x,
    reduction,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
    )


# cosine embedding loss
@st.composite
def _cos_embd_loss_helper(draw):
    dtype_inputs_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=2,
            ret_shape=True,
            num_arrays=2,
        )
    )

    input_dtypes, inputs, shape = dtype_inputs_shape

    _, label = draw(
        helpers.dtype_and_values(
            dtype=input_dtypes, shape=(shape[0],), min_value=-1, max_value=1
        ),
    )

    return input_dtypes, inputs, label


@handle_frontend_test(
    fn_tree="paddle.nn.functional.cosine_embedding_loss",
    dtype_xs_label=_cos_embd_loss_helper(),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_paddle_cosine_embedding_loss(
    *,
    dtype_xs_label,
    margin,
    reduction,
    test_flags,
    fn_tree,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs, label = dtype_xs_label
    input1_dtype, input1 = input_dtypes[0], xs[0]
    input2_dtype, input2 = input_dtypes[1], xs[1]

    helpers.test_frontend_function(
        input_dtypes=[input1_dtype, input2_dtype],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input1=input1,
        input2=input2,
        label=label[0],
        margin=margin,
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.hinge_embedding_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
)
def test_paddle_hinge_embedding_loss(
    dtype_and_x,
    margin,
    reduction,
    test_flags,
    backend_fw,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        margin=margin,
        reduction=reduction,
    )


# log_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.log_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
    ),
    epsilon=st.floats(
        min_value=1e-7,
        max_value=1.0,
    ),
)
def test_paddle_log_loss(
    dtype_and_x,
    epsilon,
    fn_tree,
    test_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        epsilon=epsilon,
    )


# smooth_l1_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.smooth_l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    delta=st.floats(
        min_value=0.1,
        max_value=1.0,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_smooth_l1_loss(
    dtype_and_x,
    delta,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
        delta=delta,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_l1_loss(
    dtype_and_x,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
    )
