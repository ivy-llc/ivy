# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_nn.test_functional.test_linear_functions import (  # noqa: E501
    x_and_linear,
)


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
    backend_fw,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        axis=axis,
    )


# Dropout2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout2d",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_dim_size=4,
        shape=(
            st.integers(min_value=2, max_value=10),
            4,
            st.integers(min_value=12, max_value=64),
            st.integers(min_value=12, max_value=64),
        ),
    ),
    p=st.floats(min_value=0.0, max_value=1.0),
    training=st.booleans(),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_dropout2d(
    *,
    d_type_and_x,
    p,
    training,
    data_format,
    backend_fw,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        p=p,
        training=training,
        data_format=data_format,
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
    backend_fw,
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
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        test_flags=test_flags,
        on_device=on_device,
        x=x[0],
        training=training,
        axis=axis,
        mode=mode,
    )


# zeropad2d
@st.composite
def _zero2pad(draw):
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            ret_shape=True,
            min_num_dims=4,
            max_num_dims=4,
            min_value=-100,
            max_value=100,
        )
    )
    ndim = len(shape)
    min_dim = min(shape)
    padding = draw(
        st.lists(
            st.integers(min_value=0, max_value=min_dim),
            min_size=ndim,
            max_size=ndim,
        )
    )
    return dtype, input, padding


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.zeropad2d",
    d_type_and_x_paddings=_zero2pad(),
    dataformat=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_zeropad2d(
    *,
    d_type_and_x_paddings,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    dataformat,
):
    dtype, x, padding = d_type_and_x_paddings
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        padding=padding,
        data_format=dataformat,
    )


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "reflect",
                "replicate",
                "circular",
            ]
        )
    )
    min_v = 4
    max_v = 4
    if mode != "constant":
        min_v = 4
        if mode == "reflect":
            max_v = 4
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            ret_shape=True,
            min_num_dims=min_v,
            max_num_dims=max_v,
            min_dim_size=4,
            max_dim_size=4,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    padding = draw(_pad_generator(shape, mode))
    if mode == "constant":
        value = draw(helpers.ints(min_value=0, max_value=4))
    else:
        value = 0.0
    return dtype, input[0], padding, value, mode


@st.composite
def _pad_generator(draw, shape, mode):
    pad = ()
    m = max(int((len(shape) + 1) / 2), 1)
    for i in range(m):
        if mode != "constant":
            if i < 2:
                max_pad_value = 0
        else:
            max_pad_value = shape[i] - 1
        pad = pad + draw(
            st.tuples(
                st.integers(min_value=0, max_value=max(0, max_pad_value)),
                st.integers(min_value=0, max_value=max(0, max_pad_value)),
            )
        )
    return pad


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.pad",
    dtype_and_input_and_other=_pad_helper(),
)
def test_paddle_pad(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, padding, value, mode = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input,
        pad=padding,
        mode=mode,
        value=value,
    )


# linear
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.linear",
    dtype_x_weight_bias=x_and_linear(
        dtypes=helpers.get_dtypes("valid", full=False),
    ),
)
def test_linear(
    *,
    dtype_x_weight_bias,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    dtype, x, weight, bias = dtype_x_weight_bias
    weight = ivy.swapaxes(weight, -1, -2)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        weight=weight,
        bias=bias,
    )
