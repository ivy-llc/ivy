# global
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.max_pool1d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=4,
        explicit_or_str_padding=True,
        return_dilation=True,
        data_format=st.sampled_from(["channel_first", "channel_last"]),
        return_data_format=True,
    ),
    ceil_mode=st.sampled_from([True, False]),
    test_gradients=st.just(False),
    ground_truth_backend="torch",
)
def test_max_pool1d(
    *,
    x_k_s_p,
    ceil_mode,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, kernel, stride, pad, dilation, data_format = x_k_s_p
    data_format = "NCW" if data_format == "channel_first" else "NWC"
    assume(not (isinstance(pad, str) and (pad.upper() == "VALID") and ceil_mode))
    # TODO: Remove this once the paddle backend supports dilation
    assume(not (backend_fw == "paddle" and max(list(dilation)) > 1))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        dilation=dilation,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.layers.max_unpool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    indices=st.lists(st.integers(0, 1), min_size=1, max_size=4),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_max_unpool1d(
    *,
    x_k_s_p,
    indices,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        indices=indices,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.max_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=2,
        max_side=4,
        explicit_or_str_padding=True,
        return_dilation=True,
        data_format=st.sampled_from(["channel_first", "channel_last"]),
        return_data_format=True,
    ),
    ceil_mode=st.sampled_from([True, False]),
    test_gradients=st.just(False),
    ground_truth_backend="jax",
)
def test_max_pool2d(
    *,
    x_k_s_p,
    ceil_mode,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, kernel, stride, pad, dilation, data_format = x_k_s_p
    assume(
        not (
            backend_fw == "tensorflow"
            and (
                (stride[0] > kernel[0] or stride[0] > kernel[1])
                or (
                    (stride[0] > 1 and dilation[0] > 1)
                    or (stride[0] > 1 and dilation[1] > 1)
                )
            )
        )
    )
    data_format = "NCHW" if data_format == "channel_first" else "NHWC"
    assume(not (isinstance(pad, str) and (pad.upper() == "VALID") and ceil_mode))
    # TODO: Remove this once the paddle backend supports dilation
    assume(not (backend_fw == "paddle" and max(list(dilation)) > 1))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        dilation=dilation,
        ceil_mode=ceil_mode,
        data_format=data_format,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.max_pool3d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=5,
        max_dims=5,
        min_side=1,
        max_side=4,
        explicit_or_str_padding=True,
        return_dilation=True,
        data_format=st.sampled_from(["channel_first", "channel_last"]),
        return_data_format=True,
    ),
    ceil_mode=st.sampled_from([True, False]),
    test_gradients=st.just(False),
    ground_truth_backend="torch",
)
def test_max_pool3d(
    *,
    x_k_s_p,
    ceil_mode,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, kernel, stride, pad, dilation, data_format = x_k_s_p

    data_format = "NCDHW" if data_format == "channel_first" else "NDHWC"
    assume(not (isinstance(pad, str) and (pad.upper() == "VALID") and ceil_mode))
    # TODO: Remove this once the paddle backend supports dilation
    assume(not (backend_fw == "paddle" and max(list(dilation)) > 1))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.avg_pool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    count_include_pad=st.booleans(),
    ceil_mode=st.booleans(),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_avg_pool1d(
    *,
    x_k_s_p,
    count_include_pad,
    ceil_mode,
    test_flags,
    backend_fw,
    on_device,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name="avg_pool1d",
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )


# avg_pool2d
@handle_test(
    fn_tree="functional.ivy.experimental.avg_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    count_include_pad=st.booleans(),
    ceil_mode=st.booleans(),
    divisor_override=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_avg_pool2d(
    *,
    x_k_s_p,
    count_include_pad,
    ceil_mode,
    divisor_override,
    data_format,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p

    if data_format == "NCHW":
        x[0] = x[0].reshape(
            (x[0].shape[0], x[0].shape[3], x[0].shape[1], x[0].shape[2])
        )

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.avg_pool3d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
    count_include_pad=st.booleans(),
    ceil_mode=st.booleans(),
    divisor_override=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_avg_pool3d(
    *,
    x_k_s_p,
    count_include_pad,
    ceil_mode,
    divisor_override,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
    )


@st.composite
def valid_dct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shared_dtype=True,
        )
    )
    dims_len = len(x[0].shape)
    n = draw(st.sampled_from([None, "int"]))
    axis = draw(helpers.ints(min_value=-dims_len, max_value=dims_len - 1))
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if n == "int":
        n = draw(helpers.ints(min_value=1, max_value=20))
        if n <= 1 and type == 1:
            n = 2
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


@handle_test(
    fn_tree="dct",
    dtype_x_and_args=valid_dct(),
    test_gradients=st.just(False),
)
def test_dct(*, dtype_x_and_args, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
    )


@handle_test(
    fn_tree="idct",
    dtype_x_and_args=valid_dct(),
    test_gradients=st.just(False),
)
def test_idct(dtype_x_and_args, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
        on_device=on_device,
    )


@st.composite
def _interp_args(draw, mode=None, mode_list=None):
    mixed_fn_compos = draw(st.booleans())
    curr_backend = ivy.current_backend_str()
    torch_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest",
        "nearest-exact",
        "area",
    ]

    tf_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest-exact",
        "tf_area",
        "bicubic_tensorflow",
        "lanczos3",
        "lanczos5",
        "mitchellcubic",
        "gaussian",
    ]

    jax_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest-exact",
        "bicubic_tensorflow",
        "lanczos3",
        "lanczos5",
    ]

    if not mode and not mode_list:
        if curr_backend == "torch" and not mixed_fn_compos:
            mode = draw(st.sampled_from(torch_modes))
        elif curr_backend == "tensorflow" and not mixed_fn_compos:
            mode = draw(st.sampled_from(tf_modes))
        elif curr_backend == "jax" and not mixed_fn_compos:
            mode = draw(st.sampled_from(jax_modes))
        else:
            mode = draw(
                st.sampled_from(
                    [
                        "linear",
                        "bilinear",
                        "trilinear",
                        "nearest",
                        "nearest-exact",
                        "area",
                        "tf_area",
                        "bicubic_tensorflow",
                        "lanczos3",
                        "lanczos5",
                        "mitchellcubic",
                        "gaussian",
                    ]
                )
            )
    elif mode_list:
        mode = draw(st.sampled_from(mode_list))
    align_corners = draw(st.one_of(st.booleans(), st.none()))
    if (curr_backend == "tensorflow" or curr_backend == "jax") and not mixed_fn_compos:
        align_corners = False
    if mode == "linear":
        num_dims = 3
    elif mode in [
        "bilinear",
        "bicubic_tensorflow",
        "bicubic",
        "mitchellcubic",
        "gaussian",
    ]:
        num_dims = 4
    elif mode == "trilinear":
        num_dims = 5
    elif mode in [
        "nearest",
        "area",
        "tf_area",
        "lanczos3",
        "lanczos5",
        "nearest-exact",
    ]:
        num_dims = (
            draw(
                helpers.ints(min_value=1, max_value=3, mixed_fn_compos=mixed_fn_compos)
            )
            + 2
        )
        align_corners = None
    if curr_backend == "tensorflow" and not mixed_fn_compos:
        num_dims = 3
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float", mixed_fn_compos=mixed_fn_compos
            ),
            min_num_dims=num_dims,
            max_num_dims=num_dims,
            min_dim_size=2,
            max_dim_size=5,
            large_abs_safety_factor=50,
            small_abs_safety_factor=50,
            safety_factor_scale="log",
        )
    )
    if draw(st.booleans()):
        scale_factor = draw(
            st.one_of(
                helpers.lists(
                    x=helpers.floats(
                        min_value=1.0, max_value=2.0, mixed_fn_compos=mixed_fn_compos
                    ),
                    min_size=num_dims - 2,
                    max_size=num_dims - 2,
                ),
                helpers.floats(
                    min_value=1.0, max_value=2.0, mixed_fn_compos=mixed_fn_compos
                ),
            )
        )
        recompute_scale_factor = draw(st.booleans())
        size = None
    else:
        size = draw(
            st.one_of(
                helpers.lists(
                    x=helpers.ints(
                        min_value=1, max_value=3, mixed_fn_compos=mixed_fn_compos
                    ),
                    min_size=num_dims - 2,
                    max_size=num_dims - 2,
                ),
                st.integers(min_value=1, max_value=3),
            )
        )
        recompute_scale_factor = False
        scale_factor = None
    if (curr_backend == "tensorflow" or curr_backend == "jax") and not mixed_fn_compos:
        if not recompute_scale_factor:
            recompute_scale_factor = True

    return (dtype, x, mode, size, align_corners, scale_factor, recompute_scale_factor)


@handle_test(
    fn_tree="functional.ivy.experimental.interpolate",
    dtype_x_mode=_interp_args(),
    antialias=st.just(False),
    test_gradients=st.just(False),
    number_positional_args=st.just(2),
)
def test_interpolate(
    dtype_x_mode, antialias, test_flags, backend_fw, fn_name, on_device
):
    (
        input_dtype,
        x,
        mode,
        size,
        align_corners,
        scale_factor,
        recompute_scale_factor,
    ) = dtype_x_mode
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-01,
        atol_=1e-01,
        x=x[0],
        size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute_scale_factor,
    )


@st.composite
def x_and_fft(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("valid", full=False))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e5,
            max_value=1e5,
            allow_inf=False,
            large_abs_safety_factor=2.5,
            small_abs_safety_factor=2.5,
            safety_factor_scale="log",
        )
    )
    dim = draw(helpers.get_axis(shape=x_dim, allow_neg=True, force_int=True))
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@handle_test(
    fn_tree="functional.ivy.experimental.fft",
    d_x_d_n_n=x_and_fft(),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_fft(*, d_x_d_n_n, test_flags, backend_fw, on_device, fn_name):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
        x=x,
        dim=dim,
        norm=norm,
        n=n,
    )


# dropout1d
@handle_test(
    fn_tree="functional.ivy.experimental.dropout1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=5,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    data_format=st.sampled_from(["NWC", "NCW"]),
    test_gradients=st.just(False),
    test_with_out=st.just(True),
)
def test_dropout1d(
    *,
    dtype_and_x,
    prob,
    training,
    data_format,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        prob=prob,
        training=training,
        data_format=data_format,
        return_flat_np_arrays=True,
    )
    ret = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
    gt_ret = helpers.flatten_and_to_np(
        backend=test_flags.ground_truth_backend, ret=gt_ret
    )
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


@handle_test(
    fn_tree="functional.ivy.experimental.dropout2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=1,
        max_dim_size=5,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
    test_gradients=st.just(False),
    test_with_out=st.just(True),
)
def test_dropout2d(
    *,
    dtype_and_x,
    prob,
    training,
    data_format,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        prob=prob,
        training=training,
        data_format=data_format,
        return_flat_np_arrays=True,
    )
    ret = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
    gt_ret = helpers.flatten_and_to_np(
        backend=test_flags.ground_truth_backend, ret=gt_ret
    )
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


# dropout3d
@handle_test(
    fn_tree="functional.ivy.experimental.dropout3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=4,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    data_format=st.sampled_from(["NCDHW", "NDHWC"]),
    test_gradients=st.just(False),
    test_with_out=st.just(True),
)
def test_dropout3d(
    *,
    dtype_and_x,
    prob,
    training,
    data_format,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        prob=prob,
        training=training,
        data_format=data_format,
        return_flat_np_arrays=True,
    )
    ret = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
    gt_ret = helpers.flatten_and_to_np(
        backend=test_flags.ground_truth_backend, ret=gt_ret
    )
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


@st.composite
def x_and_ifft(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("complex"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e-10,
            max_value=1e10,
        )
    )
    dim = draw(st.integers(1 - len(list(x_dim)), len(list(x_dim)) - 1))
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@handle_test(
    fn_tree="functional.ivy.experimental.ifft",
    d_x_d_n_n=x_and_ifft(),
    test_gradients=st.just(False),
)
def test_ifft(*, d_x_d_n_n, test_flags, backend_fw, fn_name):
    dtype, x, dim, norm, n = d_x_d_n_n

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        dim=dim,
        norm=norm,
        n=n,
    )


# embedding
@handle_test(
    fn_tree="functional.ivy.experimental.embedding",
    dtypes_indices_weights=helpers.embedding_helper(),
    max_norm=st.one_of(st.none(), st.floats(min_value=1, max_value=5)),
    number_positional_args=st.just(2),
)
def test_embedding(
    *, dtypes_indices_weights, max_norm, test_flags, backend_fw, on_device, fn_name
):
    dtypes, indices, weights, _ = dtypes_indices_weights
    dtypes = [dtypes[1], dtypes[0]]

    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        xs_grad_idxs=[[0, 0]],
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        weights=weights,
        indices=indices,
        max_norm=max_norm,
    )


@handle_test(
    fn_tree="dft",
    d_xfft_axis_n_length=x_and_fft(),
    d_xifft_axis_n_length=x_and_ifft(),
    inverse=st.booleans(),
    onesided=st.booleans(),
)
def test_dft(
    *,
    d_xfft_axis_n_length,
    d_xifft_axis_n_length,
    inverse,
    onesided,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    if inverse:
        dtype, x, axis, norm, dft_length = d_xifft_axis_n_length
    else:
        dtype, x, axis, norm, dft_length = d_xfft_axis_n_length

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        axis=axis,
        inverse=inverse,
        onesided=onesided,
        dft_length=dft_length,
        norm=norm,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.adaptive_max_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=1,
        # Setting max and min value because this operation in paddle is not
        # numerically stable
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_adaptive_max_pool2d(
    *, dtype_and_x, output_size, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        input=x[0],
        output_size=output_size,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.adaptive_avg_pool1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_value=100,
        min_value=-100,
    ),
    output_size=helpers.ints(min_value=1, max_value=5),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_adaptive_avg_pool1d(
    *, dtype_and_x, output_size, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=x[0],
        output_size=output_size,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.adaptive_avg_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=1,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_adaptive_avg_pool2d(
    *, dtype_and_x, output_size, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        input=x[0],
        output_size=output_size,
    )


@st.composite
def _reduce_window_helper(draw, get_func_st):
    dtype = draw(helpers.get_dtypes("valid", full=False))
    py_func = draw(get_func_st(dtype[0]))
    init_value = draw(
        helpers.dtype_and_values(
            dtype=dtype,
            shape=(),
            allow_inf=True,
        )
    )[1]
    ndim = draw(st.integers(min_value=1, max_value=4))
    _, others = draw(
        helpers.dtype_and_values(
            num_arrays=4,
            dtype=["int64"] * 4,
            shape=(ndim,),
            min_value=1,
            max_value=3,
            small_abs_safety_factor=1,
            large_abs_safety_factor=1,
        )
    )
    others = [other.tolist() for other in others]
    window, dilation = others[0], others[2]
    op_shape = []
    for i in range(ndim):
        min_x = window[i] + (window[i] - 1) * (dilation[i] - 1)
        op_shape.append(draw(st.integers(min_x, min_x + 1)))
    dtype, operand = draw(
        helpers.dtype_and_values(
            dtype=dtype,
            shape=op_shape,
        )
    )
    padding = draw(
        st.one_of(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=3),
                    st.integers(min_value=0, max_value=3),
                ),
                min_size=ndim,
                max_size=ndim,
            ),
            st.sampled_from(["SAME", "VALID"]),
        )
    )
    for i, arg in enumerate(others):
        if len(np.unique(arg)) == 1 and draw(st.booleans()):
            others[i] = arg[0]
    return dtype * 2, operand, init_value, py_func, others, padding


def _get_reduce_func(dtype):
    if dtype == "bool":
        return st.sampled_from([ivy.logical_and, ivy.logical_or])
    else:
        return st.sampled_from([ivy.add, ivy.maximum, ivy.minimum, ivy.multiply])


@handle_test(
    fn_tree="functional.ivy.experimental.reduce_window",
    all_args=_reduce_window_helper(_get_reduce_func),
    test_with_out=st.just(False),
    ground_truth_backend="jax",
)
def test_reduce_window(*, all_args, test_flags, backend_fw, fn_name, on_device):
    dtypes, operand, init_value, computation, others, padding = all_args
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        operand=operand[0],
        init_value=init_value[0],
        computation=computation,
        window_dimensions=others[0],
        window_strides=others[1],
        padding=padding,
        base_dilation=others[2],
        window_dilation=None,
    )


@st.composite
def x_and_fft2(draw):
    min_fft2_points = 2
    dtype = draw(helpers.get_dtypes("float_and_complex", full=False))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=2, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e5,
            max_value=1e5,
            allow_inf=False,
            large_abs_safety_factor=2.5,
            small_abs_safety_factor=2.5,
            safety_factor_scale="log",
        )
    )
    s = (
        draw(st.integers(min_fft2_points, 256)),
        draw(st.integers(min_fft2_points, 256)),
    )
    dim = draw(st.sampled_from([(0, 1), (-1, -2), (1, 0)]))
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    return dtype, x, s, dim, norm


@handle_test(
    fn_tree="functional.ivy.experimental.fft2",
    d_x_d_s_n=x_and_fft2(),
    ground_truth_backend="numpy",
    container_flags=st.just([False]),
    test_gradients=st.just(False),
)
def test_fft2(*, d_x_d_s_n, test_flags, backend_fw, fn_name, on_device):
    dtype, x, s, dim, norm = d_x_d_s_n
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        # rtol_=1e-2,
        # atol_=1e-2,
        x=x,
        s=s,
        dim=dim,
        norm=norm,
    )


@st.composite
def x_and_ifftn(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("complex"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e-10,
            max_value=1e10,
        )
    )
    axes = draw(
        st.lists(
            st.integers(0, len(x_dim) - 1), min_size=1, max_size=len(x_dim), unique=True
        )
    )
    norm = draw(st.sampled_from(["forward", "ortho", "backward"]))

    # Shape for s can be larger, smaller or equal to the size of the input
    # along the axes specified by axes.
    # Here, we're generating a list of integers corresponding to each axis in axes.
    s = draw(
        st.lists(
            st.integers(min_fft_points, 256), min_size=len(axes), max_size=len(axes)
        )
    )

    return dtype, x, s, axes, norm


@handle_test(
    fn_tree="functional.ivy.experimental.ifftn",
    d_x_d_s_n=x_and_ifftn(),
    ground_truth_backend="numpy",
    test_gradients=st.just(False),
)
def test_ifftn(
    *,
    d_x_d_s_n,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, axes, norm, s = d_x_d_s_n
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x,
        s=s,
        axes=axes,
        norm=norm,
    )


@st.composite
def x_and_rfftn(draw):
    min_rfftn_points = 2
    dtype = draw(helpers.get_dtypes("float"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=3
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e10,
            max_value=1e10,
            large_abs_safety_factor=2.5,
            small_abs_safety_factor=2.5,
            safety_factor_scale="log",
        )
    )
    axes = draw(
        st.lists(
            st.integers(0, len(x_dim) - 1), min_size=1, max_size=len(x_dim), unique=True
        )
    )
    s = draw(
        st.lists(
            st.integers(min_rfftn_points, 256), min_size=len(axes), max_size=len(axes)
        )
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    return dtype, x, s, axes, norm


@handle_test(
    fn_tree="functional.ivy.experimental.rfftn",
    d_x_d_s_n=x_and_rfftn(),
    ground_truth_backend="numpy",
    test_gradients=st.just(False),
)
def test_rfftn(
    *,
    d_x_d_s_n,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, s, axes, norm = d_x_d_s_n
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        s=s,
        axes=axes,
        norm=norm,
    )
