# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import math


def is_same_padding(padding, stride, kernel_size, input_shape):
    output_shape = tuple(
        [
            (input_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
            for i in range(len(padding))
        ]
    )
    return all(
        [
            output_shape[i] == math.ceil(input_shape[i] / stride[i])
            for i in range(len(padding))
        ]
    )


def calculate_same_padding(kernel_size, stride, shape):
    padding = tuple(
        [
            max(
                0,
                math.ceil(((shape[i] - 1) * stride[i] + kernel_size[i] - shape[i]) / 2),
            )
            for i in range(len(kernel_size))
        ]
    )
    if all([kernel_size[i] / 2 >= padding[i] for i in range(len(kernel_size))]):
        if is_same_padding(padding, stride, kernel_size, shape):
            return padding
    return (0, 0)


# avg_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.avg_pool1d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=3,
    ),
    test_with_out=st.just(False),
)
def test_torch_avg_pool1d(
    dtype_x_k_s,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, padding = dtype_x_k_s

    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], x[0].shape[1]))
    x_shape = list(x[0].shape[2])

    # Torch ground truth func also takes padding input as an integer
    # or a tuple of integers, not a string
    padding = tuple(
        [
            ivy.handle_padding(x_shape[i], stride[0], kernel_size[i], padding)
            for i in range(len(x_shape))
        ]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=True,
    )


# avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.avg_pool2d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
    ),
    ceil_mode=st.booleans(),
    count_include_pad=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_avg_pool2d(
    dtype_x_k_s,
    count_include_pad,
    ceil_mode,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, pad_name = dtype_x_k_s

    if len(stride) == 1:
        stride = (stride[0], stride[0])

    if pad_name == "SAME":
        padding = calculate_same_padding(kernel_size, stride, x[0].shape[2:])
    else:
        padding = (0, 0)

    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.avg_pool3d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=5,
        max_dims=5,
        min_side=1,
        max_side=4,
        data_format="channel_first",
        only_explicit_padding=True,
    ),
    count_include_pad=st.booleans(),
    ceil_mode=st.booleans(),
    divisor_override=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
    test_with_out=st.just(False),
)
def test_torch_avg_pool3d(
    *,
    dtype_x_k_s,
    count_include_pad,
    ceil_mode,
    divisor_override,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, padding = dtype_x_k_s

    padding = [padding[i][0] for i in range(len(padding))]

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


# max_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.max_pool1d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=3,
        explicit_or_str_padding=False,
        only_explicit_padding=True,
    ),
    test_with_out=st.just(False),
)
def test_torch_max_pool1d(
    dtype_x_k_s,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, padding = dtype_x_k_s

    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], x[0].shape[1]))
    x_shape = [x[0].shape[2]]

    # Torch ground truth func also takes padding input as an integer
    # or a tuple of integers, not a string
    padding = tuple(
        [
            ivy.functional.layers._handle_padding(
                x_shape[i], stride[0], kernel_size[i], padding
            )
            for i in range(len(x_shape))
        ]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


# max_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.max_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
        explicit_or_str_padding=True,
        return_dilation=True,
    ).filter(lambda x: x[4] != "VALID" and x[4] != "SAME"),
    test_with_out=st.just(False),
    ceil_mode=st.just(True),
)
def test_torch_max_pool2d(
    x_k_s_p,
    ceil_mode,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x, kernel, stride, pad, dilation = x_k_s_p
    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCHW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))
    pad = (pad[0][0], pad[1][0])

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel,
        stride=stride,
        padding=pad,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


# adaptive_avg_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=helpers.ints(min_value=1, max_value=10),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool1d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
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
        output_size=output_size,
        atol=1e-2,
    )


# adaptive_avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=10),
            helpers.ints(min_value=1, max_value=10),
        ),
        helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool2d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
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
        output_size=output_size,
        atol=1e-2,
    )


# adaptive_max_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_max_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=5,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=10),
            helpers.ints(min_value=1, max_value=10),
        ),
        helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_adaptive_max_pool2d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
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
        output_size=output_size,
        atol=1e-2,
    )


# avg_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.lp_pool1d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=3,
    ),
    norm_type=helpers.number(min_value=0.1, max_value=6),
    test_with_out=st.just(False),
)
def test_torch_lp_pool1d(
    dtype_x_k_s,
    norm_type,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, _ = dtype_x_k_s

    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], x[0].shape[1]))

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        norm_type=norm_type if norm_type > 0 else 1,
        kernel_size=kernel_size[0],
        stride=stride[0],
        ceil_mode=False,
    )


# avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.lp_pool2d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
    ),
    norm_type=helpers.number(min_value=0.1, max_value=6),
    test_with_out=st.just(False),
)
def test_torch_lp_pool2d(
    dtype_x_k_s,
    norm_type,
    *,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, _ = dtype_x_k_s
    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCW
    x[0] = x[0].transpose((0, 3, 1, 2))

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        norm_type=norm_type if norm_type > 0 else 1,
        kernel_size=kernel_size,
        stride=stride[0],
        ceil_mode=False,
    )
