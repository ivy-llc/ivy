# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.nn.functional.binary_cross_entropy_with_logits",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1.0013580322265625e-05,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=1,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_binary_cross_entropy_with_logits(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    # TODO: paddle's implementation of pos_weight is wrong
    # https://github.com/PaddlePaddle/Paddle/blob/f0422a28d75f9345fa3b801c01cd0284b3b44be3/python/paddle/nn/functional/loss.py#L831
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            weight_dtype[0],
        ],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logit=x[0],
        label=x[0],
        weight=weight[0],
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
    frontend,
    on_device,
):
    input_dtypes, xs, label = dtype_xs_label
    input1_dtype, input1 = input_dtypes[0], xs[0]
    input2_dtype, input2 = input_dtypes[1], xs[1]

    helpers.test_frontend_function(
        input_dtypes=[input1_dtype, input2_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input1=input1,
        input2=input2,
        label=label[0],
        margin=margin,
        reduction=reduction,
    )
