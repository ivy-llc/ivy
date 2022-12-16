# global
from hypothesis import strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.experimental.max_pool2d",
    ground_truth_backend="jax",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
)
def test_max_pool2d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.experimental.max_pool1d",
    ground_truth_backend="jax",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
)
def test_max_pool1d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.experimental.avg_pool1d",
    ground_truth_backend="jax",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
)
def test_avg_pool1d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name="avg_pool1d",
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.experimental.max_pool3d",
    ground_truth_backend="jax",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
)
def test_max_pool3d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.experimental.avg_pool3d",
    ground_truth_backend="jax",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
)
def test_avg_pool3d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_test(
    fn_tree="functional.experimental.avg_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
)
def test_avg_pool2d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@st.composite
def valid_dct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
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
    axis = draw(helpers.ints(min_value=-dims_len, max_value=dims_len))
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
)
def test_dct(
    dtype_x_and_args,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
    )


@st.composite
def x_and_fft(draw, dtypes):
    min_fft_points = 2
    dtype = draw(dtypes)
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
        )
    )
    dim = draw(
        helpers.get_axis(shape=x_dim, allow_neg=True, allow_none=False, max_size=1)
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@handle_test(
    fn_tree="functional.ivy.fft",
    d_x_d_n_n=x_and_fft(helpers.get_dtypes("complex")),
    ground_truth_backend="numpy",
)
def test_fft(
    *,
    d_x_d_n_n,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=False,
        x=x,
        dim=dim,
        norm=norm,
        n=n,
    )


# dropout1d
@handle_test(
    fn_tree="functional.ivy.dropout1d",
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
)
def test_dropout1d(
    *,
    dtype_and_x,
    prob,
    training,
    data_format,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    backend_fw,
    on_device,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
        prob=prob,
        training=training,
        data_format=data_format,
        return_flat_np_arrays=True,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    gt_ret = helpers.flatten_and_to_np(ret=gt_ret)
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape
