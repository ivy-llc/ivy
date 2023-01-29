# global
from hypothesis import strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.max_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=4, max_dims=4, min_side=1, max_side=4, allow_explicit_padding=True
    ),
    test_gradients=st.just(False),
    container_flags=st.just([False]),
)
def test_max_pool2d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.max_pool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    test_gradients=st.just(False),
)
def test_max_pool1d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.avg_pool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    test_gradients=st.just(False),
)
def test_avg_pool1d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.max_pool3d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
    test_gradients=st.just(False),
)
def test_max_pool3d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.avg_pool3d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
    test_gradients=st.just(False),
)
def test_avg_pool3d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.avg_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    test_gradients=st.just(False),
)
def test_avg_pool2d(
    *,
    x_k_s_p,
    test_flags,
    backend_fw,
    fn_name,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        ground_truth_backend="jax",
        input_dtypes=dtype,
        test_flags=test_flags,
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
    test_gradients=st.just(False),
)
def test_dct(
    dtype_x_and_args,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
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
    fn_tree="functional.ivy.experimental.fft",
    d_x_d_n_n=x_and_fft(helpers.get_dtypes("complex")),
    ground_truth_backend="numpy",
    test_gradients=st.just(False),
)
def test_fft(
    *,
    d_x_d_n_n,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
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
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
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
def test_ifft(
    *,
    d_x_d_n_n,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x, dim, norm, n = d_x_d_n_n

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    *,
    dtypes_indices_weights,
    max_norm,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
    ground_truth_backend,
):
    dtypes, indices, weights, _ = dtypes_indices_weights
    dtypes = [dtypes[1], dtypes[0]]

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtypes,
        test_flags=test_flags,
        xs_grad_idxs=[[0, 0]],
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        weights=weights,
        indices=indices,
        max_norm=max_norm,
    )


@handle_test(
    fn_tree="dft",
    d_xfft_axis_n_length=x_and_fft(helpers.get_dtypes("complex")),
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
    ground_truth_backend,
):
    if inverse:
        dtype, x, axis, norm, dft_length = d_xifft_axis_n_length
    else:
        dtype, x, axis, norm, dft_length = d_xfft_axis_n_length

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        x=x,
        axis=axis,
        inverse=inverse,
        onesided=onesided,
        dft_length=dft_length,
        norm=norm,
        rtol_=1e-2,
        atol_=1e-2,
    )
