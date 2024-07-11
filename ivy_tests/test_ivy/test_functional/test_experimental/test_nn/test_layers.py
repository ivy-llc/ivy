# global
import numpy as np
import torch
from hypothesis import strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, BackendHandler


# --- Helpers --- #
# --------------- #


def _get_reduce_func(dtype):
    if dtype == "bool":
        return st.sampled_from([ivy.logical_and, ivy.logical_or])
    else:
        return st.sampled_from([ivy.add, ivy.maximum, ivy.minimum, ivy.multiply])


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
        "bicubic",
    ]
    tf_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest-exact",
        "tf_area",
        "tf_bicubic",
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
        "tf_bicubic",
        "lanczos3",
        "lanczos5",
    ]
    if mode_list == "torch":
        mode_list = torch_modes
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
                        "tf_bicubic",
                        "lanczos3",
                        "lanczos5",
                        "mitchellcubic",
                        "gaussian",
                    ]
                )
            )
    elif mode_list:
        mode = draw(st.sampled_from(mode_list))
    if mode == "linear":
        num_dims = 3
    elif mode in [
        "bilinear",
        "tf_bicubic",
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
            max_value=1e04,
            min_value=-1e04,
            abs_smallest_val=1e-04,
        )
    )
    align_corners = draw(st.booleans())
    if draw(st.booleans()):
        if draw(st.booleans()):
            scale_factor = draw(
                st.floats(min_value=max([1 / d for d in x[0].shape[2:]]), max_value=3)
            )
        else:
            scale_factor = []
            for s in x[0].shape[2:]:
                scale_factor += [draw(st.floats(min_value=1 / s, max_value=3))]
        recompute_scale_factor = draw(st.booleans())
        size = None
    else:
        size = draw(
            st.one_of(
                st.lists(
                    st.integers(min_value=1, max_value=3 * max(x[0].shape)),
                    min_size=num_dims - 2,
                    max_size=num_dims - 2,
                ),
                st.integers(min_value=1, max_value=3 * max(x[0].shape)),
            )
        )
        recompute_scale_factor = None
        scale_factor = None
    return (dtype, x, mode, size, align_corners, scale_factor, recompute_scale_factor)


@st.composite
def _lstm_helper(draw):
    input_size = draw(helpers.ints(min_value=2, max_value=5))
    hidden_size = 4 * input_size
    input_length = draw(helpers.ints(min_value=2, max_value=5))
    batch_size = draw(helpers.ints(min_value=1, max_value=4)) * 2
    dtype = draw(helpers.get_dtypes("float", full=False))
    (time_major, go_backwards, unroll, zero_output_for_mask, return_all_outputs) = draw(
        helpers.array_bools(size=5)
    )
    shape = [batch_size, input_length, input_size]
    if time_major:
        shape = [input_length, batch_size, input_size]
    inputs = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            shape=shape,
            min_value=-1,
            max_value=1,
            abs_smallest_val=1e-5,
            safety_factor_scale="log",
        )
    )[1][0]
    mask = draw(
        st.just([None, [None]])
        | helpers.dtype_and_values(
            available_dtypes=["bool"],
            shape=[*shape[:2], 1],
        )
    )[1][0]
    kernel, recurrent_kernel = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            num_arrays=2,
            shape=(input_size, hidden_size),
            min_value=-1,
            max_value=1,
            abs_smallest_val=1e-5,
            safety_factor_scale="log",
        )
    )[1]
    bias, recurrent_bias = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            num_arrays=2,
            shape=(1, hidden_size),
            min_value=-1,
            max_value=1,
            abs_smallest_val=1e-5,
            safety_factor_scale="log",
        )
    )[1]
    init_h, init_c = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            num_arrays=2,
            shape=(batch_size, input_size),
            min_value=-1,
            max_value=1,
            abs_smallest_val=1e-5,
            safety_factor_scale="log",
        )
    )[1]
    dtypes = [dtype[0] for _ in range(7)]
    if mask is not None:
        dtypes.append("bool")
    # ToDo : zero_output_for_mask doesn't work if we don't return_all_outputs
    # in tensorflow
    zero_output_for_mask = zero_output_for_mask and return_all_outputs
    return (
        dtypes,
        inputs,
        kernel,
        recurrent_kernel,
        bias,
        recurrent_bias,
        [init_h, init_c],
        go_backwards,
        mask,
        unroll,
        input_length,
        time_major,
        zero_output_for_mask,
        return_all_outputs,
    )


@st.composite
def _reduce_window_helper(draw, get_func_st):
    dtype = draw(helpers.get_dtypes("valid", full=False, index=2))
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


@st.composite
def _valid_dct(draw):
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


@st.composite
def _valid_stft(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    frame_length = draw(helpers.ints(min_value=16, max_value=100))
    frame_step = draw(helpers.ints(min_value=1, max_value=50))

    return dtype, x, frame_length, frame_step


@st.composite
def _x_and_fft(draw):
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


@st.composite
def _x_and_fft2(draw):
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


@st.composite
def _x_and_ifft(draw):
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


@st.composite
def _x_and_ifftn(draw):
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


@st.composite
def _x_and_ifftn_jax(draw):
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
            st.integers(0, len(x_dim) - 1),
            min_size=1,
            max_size=min(len(x_dim), 3),
            unique=True,
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


@st.composite
def _x_and_rfft(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("numeric"))
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
    axis = draw(st.integers(1 - len(list(x_dim)), len(list(x_dim)) - 1))
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, axis, norm, n


@st.composite
def _x_and_rfftn(draw):
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


@st.composite
def max_unpool1d_helper(
    draw,
    **data_gen_kwargs,
):
    dts, values, kernel_size, strides, _ = draw(
        helpers.arrays_for_pooling(
            min_dims=3,
            max_dims=3,
            data_format="channel_first",
            **data_gen_kwargs,
        )
    )
    dts.extend(["int64"])
    values = values[0]
    if dts[0] in ["float16", "bfloat16"]:
        values = values.astype(np.float32)
        dts[0] = "float32"
    padding = draw(helpers.ints(min_value=0, max_value=2))
    if padding > (kernel_size[0] // 2):
        padding = 0

    values, indices = torch.nn.functional.max_pool1d(
        torch.tensor(values.astype(np.float32)),
        kernel_size,
        strides,
        padding,
        return_indices=True,
    )
    indices = indices.numpy().astype(np.int64)
    max_idx = values.shape[-1] - 1
    indices = np.where(indices > max_idx, max_idx, indices)
    values = values.numpy().astype(dts[0])
    return dts, values, indices, kernel_size, strides, padding


# --- Main --- #
# ------------ #


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
        min_dim_size=2,
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
    data_format=st.sampled_from(["NCHW", "NHWC"]),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_adaptive_avg_pool2d(
    *, dtype_and_x, output_size, data_format, test_flags, backend_fw, fn_name, on_device
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
        data_format=data_format,
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
    fn_tree="functional.ivy.experimental.adaptive_max_pool3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_num_dims=5,
        min_dim_size=1,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_adaptive_max_pool3d(
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


@handle_test(
    fn_tree="dct",
    dtype_x_and_args=_valid_dct(),
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
    fn_tree="dft",
    d_xfft_axis_n_length=_x_and_fft(),
    d_xifft_axis_n_length=_x_and_ifft(),
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
    fn_tree="functional.ivy.experimental.fft",
    d_x_d_n_n=_x_and_fft(),
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


@handle_test(
    fn_tree="functional.ivy.experimental.fft2",
    d_x_d_s_n=_x_and_fft2(),
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
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        s=s,
        dim=dim,
        norm=norm,
    )


@handle_test(
    fn_tree="idct",
    dtype_x_and_args=_valid_dct(),
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


@handle_test(
    fn_tree="functional.ivy.experimental.ifft",
    d_x_d_n_n=_x_and_ifft(),
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


@handle_test(
    fn_tree="functional.ivy.experimental.ifftn",
    d_x_d_s_n=_x_and_ifftn(),
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
    dtype, x, s, axes, norm = d_x_d_s_n
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


@handle_test(
    fn_tree="functional.ivy.experimental.interpolate",
    dtype_x_mode=_interp_args(),
    test_gradients=st.just(False),
    number_positional_args=st.just(2),
)
def test_interpolate(dtype_x_mode, test_flags, backend_fw, fn_name, on_device):
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
        atol_=1e-03,
        x=x[0],
        size=size,
        mode=mode,
        align_corners=align_corners,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute_scale_factor,
    )


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
    assume(backend_fw != "paddle" or max(list(dilation)) <= 1)

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
            and all(
                stride[i] > kernel[i] or (stride[i] > 1 and dilation[i] > 1)
                for i in range(2)
            )
        )
    )
    data_format = "NCHW" if data_format == "channel_first" else "NHWC"
    assume(not (isinstance(pad, str) and (pad.upper() == "VALID") and ceil_mode))
    # TODO: Remove this once the paddle backend supports dilation
    assume(backend_fw != "paddle" or max(list(dilation)) <= 1)

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
    assume(
        not (
            backend_fw == "tensorflow"
            and isinstance(pad, str)
            and pad == "SAME"
            and any(dil > 1 for dil in dilation)
        )
    )
    data_format = "NCDHW" if data_format == "channel_first" else "NDHWC"
    assume(not (isinstance(pad, str) and (pad.upper() == "VALID") and ceil_mode))
    # TODO: Remove this once the paddle backend supports dilation
    assume(backend_fw != "paddle" or max(list(dilation)) <= 1)

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
    fn_tree="functional.ivy.experimental.max_unpool1d",
    x_k_s_p=max_unpool1d_helper(min_side=2, max_side=5),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_max_unpool1d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, ind, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        input=x,
        indices=ind,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
    )


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


@handle_test(
    fn_tree="functional.ivy.experimental.rfft",
    dtype_x_axis_norm_n=_x_and_rfft(),
    ground_truth_backend="numpy",
)
def test_rfft(
    *,
    dtype_x_axis_norm_n,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, axis, norm, n = dtype_x_axis_norm_n
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        n=n,
        axis=axis,
        norm=norm,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.rfftn",
    d_x_d_s_n=_x_and_rfftn(),
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


# test_rnn
@handle_test(
    fn_tree="functional.ivy.experimental.rnn",
    rnn_args=_lstm_helper(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
)
def test_rnn(
    *,
    rnn_args,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    # ToDo : Get the tests passing with paddle
    (
        input_dtypes,
        inputs,
        kernel_orig,
        recurrent_kernel_orig,
        bias_orig,
        recurrent_bias_orig,
        initial_states,
        go_backwards,
        mask,
        unroll,
        input_length,
        time_major,
        zero_output_for_mask,
        return_all_outputs,
    ) = rnn_args

    # unsupported dtype of float16 is in our _lstm_step function
    # so can't be inferred through ivy.function_unsupported_devices_and_dtypes
    assume(not (backend_fw == "torch" and input_dtypes[0] == "float16"))

    def _lstm_step(cell_inputs, cell_states):
        with BackendHandler.update_backend(
            ivy.current_backend(
                cell_inputs.to_native()
                if "ivy" in str(type(cell_inputs))
                else cell_inputs
            ).backend
        ) as ivy_backend:
            nonlocal kernel_orig, recurrent_kernel_orig, bias_orig, recurrent_bias_orig
            kernel = ivy_backend.array(kernel_orig)
            recurrent_kernel = ivy_backend.array(recurrent_kernel_orig)
            bias = ivy_backend.array(bias_orig)
            recurrent_bias = ivy_backend.array(recurrent_bias_orig)

            h_tm1 = cell_states[0]  # previous memory state
            c_tm1 = cell_states[1]  # previous carry state

            z = ivy_backend.dot(cell_inputs, kernel) + bias
            z += ivy_backend.dot(h_tm1, recurrent_kernel) + recurrent_bias

            z0, z1, z2, z3 = ivy_backend.split(z, num_or_size_splits=4, axis=-1)

            i = ivy_backend.sigmoid(z0)  # input
            f = ivy_backend.sigmoid(z1)  # forget
            c = f * c_tm1 + i * ivy_backend.tanh(z2)
            o = ivy_backend.sigmoid(z3)  # output

            h = o * ivy_backend.tanh(c)
            return h, [h, c]

    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        step_function=_lstm_step,
        inputs=inputs,
        initial_states=initial_states,
        go_backwards=go_backwards,
        mask=mask,
        constants=None,
        unroll=unroll,
        input_length=input_length,
        time_major=time_major,
        zero_output_for_mask=zero_output_for_mask,
        return_all_outputs=return_all_outputs,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.sliding_window",
    all_args=helpers.arrays_for_pooling(3, 3, 1, 2, return_dilation=True),
    test_with_out=st.just(False),
    ground_truth_backend="jax",
)
def test_sliding_window(*, all_args, test_flags, backend_fw, fn_name, on_device):
    dtypes, input, k, stride, padding, dilation = all_args
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        input=input,
        window_size=k,
        stride=stride[0],
        dilation=dilation[0],
        padding=padding,
    )


# test_stft
@handle_test(
    fn_tree="functional.ivy.experimental.stft",
    dtype_x_and_args=_valid_stft(),
    ground_truth_backend="tensorflow",
    test_gradients=st.just(False),
)
def test_stft(
    *,
    dtype_x_and_args,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x, frame_length, frame_step = dtype_x_and_args
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        signals=x[0],
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=None,
        window_fn=None,
        pad_end=True,
    )
