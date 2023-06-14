# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
