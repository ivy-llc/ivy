# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _batch_norm_helper(draw):
    num_dims = draw(st.integers(min_value=4, max_value=5))
    dtype, x = draw(helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=num_dims,
        max_num_dims=num_dims,
        min_value=-1e02,
        max_value=1e02,
    ))
    epsilon = draw(st.floats(min_value=1e-07, max_value=1e-04))
    factor = draw(st.floats(min_value=0.5, max_value=1))
    training = draw(st.booleans())
    if num_dims == 4:
        data_format = draw(st.sampled_from(["NHWC", "NCHW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))
    num_channels = x[0].shape[data_format.rfind("C")]
    dtypes, vectors = draw(helpers.dtype_and_values(
        available_dtypes=["float32"],
        shape=(num_channels,),
        num_arrays=4,
        min_value=-1e02,
        max_value=1e02,
    ))
    vectors[3] = np.abs(vectors[3])  # non-negative variance
    return dtype + dtypes, x, epsilon, factor, training, data_format, vectors


@handle_frontend_test(
    fn_tree="tensorflow.compat.v1.nn.fused_batch_norm",
    dtypes_args=_batch_norm_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_fused_batch_norm(
    *,
    dtypes_args,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    dtypes, x, epsilon, factor, training, data_format, vectors = dtypes_args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-02,
        x=x[0],
        scale=vectors[0],
        offset=vectors[1],
        mean=vectors[2],
        variance=vectors[3],
        epsilon=epsilon,
        data_format=data_format,
        is_training=training,
        exponential_avg_factor=factor,
    )
