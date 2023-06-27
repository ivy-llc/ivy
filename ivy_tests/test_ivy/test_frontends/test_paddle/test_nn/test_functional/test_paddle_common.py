# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Cosine Similarity
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.cosine_similarity",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_dim_size=2,
        shape=(4, 4),
    ),
    axis=st.integers(min_value=-1, max_value=1),
)
def test_paddle_cosine_similarity(
    *,
    d_type_and_x,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        axis=axis,
    )


# dropout
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    p=st.floats(min_value=0.0, max_value=1.0),
    axis=st.integers(min_value=0, max_value=1),
    training=st.booleans(),
    mode=st.one_of(
        *[st.just(seq) for seq in ["upscale_in_train", "downscale_in_infer"]]
    ),
)
def test_paddle_dropout(
    *,
    d_type_and_x,
    p,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    training,
    axis,
    mode,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        p=p,
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
        on_device=on_device,
        x=x[0],
        training=training,
        axis=axis,
        mode=mode,
    )
