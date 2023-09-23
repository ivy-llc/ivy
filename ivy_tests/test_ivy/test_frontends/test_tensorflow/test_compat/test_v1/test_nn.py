# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_nn import _x_and_filters


# --- Helpers --- #
# --------------- #


@st.composite
def _batch_norm_helper(draw):
    num_dims = draw(st.integers(min_value=4, max_value=5))
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=num_dims,
            max_num_dims=num_dims,
            min_value=-1e02,
            max_value=1e02,
        )
    )
    epsilon = draw(st.floats(min_value=1e-07, max_value=1e-04))
    factor = draw(st.floats(min_value=0.5, max_value=1))
    training = draw(st.booleans())
    if num_dims == 4:
        data_format = draw(st.sampled_from(["NHWC", "NCHW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))
    num_channels = x[0].shape[data_format.rfind("C")]
    dtypes, vectors = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32"],
            shape=(num_channels,),
            num_arrays=4,
            min_value=-1e02,
            max_value=1e02,
        )
    )
    vectors[3] = np.abs(vectors[3])  # non-negative variance
    return dtype + dtypes, x, epsilon, factor, training, data_format, vectors


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="tensorflow.compat.v1.nn.depthwise_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="depthwise",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_depthwise_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filter=filters,
        strides=stride,
        padding=padding,
        rate=dilation,
        name=None,
        data_format=data_format,
    )


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
    backend_fw,
    fn_tree,
    on_device,
):
    dtypes, x, epsilon, factor, training, data_format, vectors = dtypes_args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
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


# max_pool
@handle_frontend_test(
    fn_tree="tensorflow.compat.v1.nn.max_pool",
    data_format=st.just("NHWC"),
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    test_with_out=st.just(False),
)
def test_tensorflow_max_pool(
    *,
    x_k_s_p,
    data_format,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )


@handle_frontend_test(
    fn_tree="tensorflow.compat.v1.nn.separable_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="separable",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_separable_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        depthwise_filter=filters[0],
        pointwise_filter=filters[1],
        strides=stride,
        padding=padding,
        rate=dilation,
        name=None,
        data_format=data_format,
    )
