"""Collection of tests for training unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

# Linear #
# -------#


# linear
@given(
    batch_shape=helpers.lists(
        st.integers(1, 2), min_size="num_dims", max_size="num_dims", size_bounds=[1, 2]
    ),
    input_channels=st.integers(2, 4),
    output_channels=st.integers(1, 5),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    with_v=st.booleans(),
    as_variable=st.booleans(),
)
def test_linear_layer_training(
    batch_shape,
    input_channels,
    output_channels,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(
            ivy.zeros(batch_shape, device=device),
            ivy.ones(batch_shape, device=device),
            input_channels,
            device=device,
        ),
        dtype=dtype,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.asarray(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, output_channels]), device=device, dtype=dtype)
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, device=device, v=v)

    def loss_fn(x_, v_):
        out = linear_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(linear_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, linear_layer.v = train_step(x, linear_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# Convolutions #
# -------------#


# conv1d
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    input_channels=st.integers(1, 4),
    filter_size=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    output_channels=st.integers(1, 5),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv1d_layer_training(
    batch_size,
    width,
    input_channels,
    filter_size,
    padding,
    output_channels,
    with_v,
    dtype,
    as_variable,
    device,
    fw,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, width, input_channels)
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        print(wlim)

        w = ivy.variable(
            ivy.asarray(
                np.random.uniform(
                    -wlim, wlim, (filter_size, input_channels, output_channels)
                ),
                dtype=dtype,
                device=device,
            )
        )

        b = ivy.variable(ivy.zeros([1, 1, output_channels]), device=device, dtype=dtype)
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None

    conv1d_layer = ivy.Conv1D(
        input_channels, output_channels, filter_size, 1, padding, device=device, v=v
    )

    def loss_fn(x_, v_):
        out = conv1d_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv1d_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv1d_layer.v = train_step(x, conv1d_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# conv1d transpose
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    input_channels=st.integers(1, 5),
    filter_size=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    output_channels=st.integers(1, 5),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv1d_transpose_layer_training(
    batch_size,
    width,
    input_channels,
    output_channels,
    filter_size,
    padding,
    with_v,
    dtype,
    as_variable,
    device,
    fw,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call is helpers.mx_call:
        # to_scalar syncrhonization issues
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, width, input_channels)
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, (filter_size, output_channels, input_channels)
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, output_channels]))
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    conv1d_trans_layer = ivy.Conv1DTranspose(
        input_channels,
        output_channels,
        filter_size,
        1,
        padding,
        output_shape=output_shape,
        device=device,
        v=v,
    )

    def loss_fn(x_, v_):
        out = conv1d_trans_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv1d_trans_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv1d_trans_layer.v = train_step(x, conv1d_trans_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# conv2d
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    height=st.integers(1, 3),
    input_channels=st.integers(1, 5),
    filter_height=st.integers(1, 7),
    filter_width=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    output_channels=st.integers(1, 5),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv2d_layer_training(
    batch_size,
    width,
    height,
    input_channels,
    filter_height,
    filter_width,
    output_channels,
    padding,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, height, width, input_channels)
    filter_shape = [filter_height, filter_width]
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels]), dtype=dtype, device=device
        )
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    conv2d_layer = ivy.Conv2D(
        input_channels, output_channels, filter_shape, 1, padding, device=device, v=v
    )

    def loss_fn(x_, v_):
        out = conv2d_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv2d_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv2d_layer.v = train_step(x, conv2d_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# conv2d transpose
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    height=st.integers(1, 3),
    input_channels=st.integers(1, 5),
    filter_width=st.integers(1, 7),
    filter_height=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    output_channels=st.integers(1, 5),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv2d_transpose_layer_training(
    batch_size,
    width,
    height,
    input_channels,
    filter_height,
    filter_width,
    padding,
    output_channels,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call is helpers.mx_call:
        # to_scalar syncrhonization issues
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, height, width, input_channels)
    filter_shape = [filter_height, filter_width]
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype="float32",
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, output_channels]), dtype="float32", device=device
        )
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    conv2d_transpose_layer = ivy.Conv2DTranspose(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        output_shape=output_shape,
        device=device,
        v=v,
    )

    def loss_fn(x_, v_):
        out = conv2d_transpose_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv2d_transpose_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv2d_transpose_layer.v = train_step(x, conv2d_transpose_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# depthwise conv2d
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    height=st.integers(1, 3),
    num_channels=st.integers(1, 5),
    filter_width=st.integers(1, 7),
    filter_height=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_depthwise_conv2d_layer_training(
    batch_size,
    width,
    height,
    num_channels,
    filter_height,
    filter_width,
    padding,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call is helpers.mx_call:
        # to_scalar syncrhonization issues
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    # smoke test
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    output_shape = (batch_size, height, width, num_channels)
    filter_shape = [filter_height, filter_width]
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (num_channels * 2)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, tuple(filter_shape + [num_channels])),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([1, 1, num_channels]), dtype=dtype, device=device)
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    depthwise_conv2d_layer = ivy.DepthwiseConv2D(
        num_channels, filter_shape, 1, padding, device=device, v=v
    )

    def loss_fn(x_, v_):
        out = depthwise_conv2d_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(depthwise_conv2d_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, depthwise_conv2d_layer.v = train_step(x, depthwise_conv2d_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# conv3d
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    height=st.integers(1, 3),
    input_channels=st.integers(1, 5),
    output_channels=st.integers(1, 5),
    filter_height=st.integers(1, 7),
    filter_width=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv3d_layer_training(
    batch_size,
    width,
    height,
    input_channels,
    output_channels,
    filter_height,
    filter_width,
    padding,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call is helpers.mx_call:
        # to_scalar syncrhonization issues
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    # smoke test
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, height, width, input_channels)
    filter_shape = [filter_height, filter_width]
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels]), dtype=dtype, device=device
        )
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    conv3d_layer = ivy.Conv3D(
        input_channels, output_channels, filter_shape, 1, padding, device=device, v=v
    )

    def loss_fn(x_, v_):
        out = conv3d_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv3d_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv3d_layer.v = train_step(x, conv3d_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# conv3d transpose
@given(
    batch_size=st.integers(1, 10),
    width=st.integers(1, 3),
    height=st.integers(1, 3),
    input_channels=st.integers(1, 5),
    filter_width=st.integers(1, 7),
    filter_height=st.integers(1, 7),
    padding=st.sampled_from(("SAME", "VALID")),
    output_channels=st.integers(1, 5),
    with_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_conv3d_transpose_layer_training(
    batch_size,
    width,
    height,
    input_channels,
    filter_width,
    filter_height,
    padding,
    output_channels,
    with_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d does not work when CUDA is installed, but array is on CPU
        return
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d
        return
    if call in [helpers.mx_call] and "cpu" in device:
        # mxnet only supports 3d transpose convolutions with CUDNN
        return
    if call in [helpers.torch_call] and (dtype == "float16"):
        # torch.nn.functional.conv1d doesn't handle float16
        return
    # smoke test
    output_shape = (batch_size, height, width, input_channels)
    filter_shape = [filter_height, filter_width]
    x = ivy.asarray(
        ivy.random_normal(shape=output_shape),
        dtype=dtype,
        device=device,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(
                    -wlim, wlim, tuple(filter_shape + [output_channels, input_channels])
                ),
                dtype=dtype,
                device=device,
            )
        )
        b = ivy.variable(
            ivy.zeros([1, 1, 1, 1, output_channels]), dtype=dtype, device=device
        )
        v = ivy.Container({"w": w, "b": b})
    else:
        v = None
    conv3d_transpose_layer = ivy.Conv3DTranspose(
        input_channels,
        output_channels,
        filter_shape,
        1,
        padding,
        output_shape=output_shape,
        device=device,
        v=v,
    )

    def loss_fn(x_, v_):
        out = conv3d_transpose_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(conv3d_transpose_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, conv3d_transpose_layer.v = train_step(x, conv3d_transpose_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# LSTM #
# -----#


# lstm
@given(
    b_t_ic_hc_otf_sctv=st.sampled_from(
        [
            (
                2,
                3,
                4,
                5,
                [0.93137765, 0.9587628, 0.96644664, 0.93137765, 0.9587628, 0.96644664],
                3.708991,
            ),
        ],
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
)
def test_lstm_layer_training(
    b_t_ic_hc_otf_sctv, with_v, dtype, as_variable, device, compile_graph, call
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    if call is helpers.mx_call:
        # to_scalar syncrhonization issues
        return
    # smoke test
    (
        b,
        t,
        input_channels,
        hidden_channels,
        output_true_flat,
        state_c_true_val,
    ) = b_t_ic_hc_otf_sctv
    x = ivy.astype(
        ivy.linspace(
            ivy.zeros([b, t], device=device),
            ivy.ones([b, t], device=device),
            input_channels,
        ),
        dtype="float32",
    )
    if with_v:
        kernel = ivy.variable(
            ivy.ones([input_channels, 4 * hidden_channels], device=device) * 0.5
        )
        recurrent_kernel = ivy.variable(
            ivy.ones([hidden_channels, 4 * hidden_channels], device=device) * 0.5
        )
        v = ivy.Container(
            {
                "input": {"layer_0": {"w": kernel}},
                "recurrent": {"layer_0": {"w": recurrent_kernel}},
            }
        )
    else:
        v = None
    lstm_layer = ivy.LSTM(input_channels, hidden_channels, device=device, v=v)

    def loss_fn(x_, v_):
        out, (state_h, state_c) = lstm_layer(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(lstm_layer.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, lstm_layer.v = train_step(x, lstm_layer.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()


# Sequential #
# -----------#


# sequential
@given(
    batch_shape=helpers.lists(
        st.integers(1, 2), min_size="num_dims", max_size="num_dims", size_bounds=[1, 2]
    ),
    channels=st.integers(1, 5),
    with_v=st.booleans(),
    seq_v=st.booleans(),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    as_variable=st.booleans(),
)
def test_sequential_layer_training(
    batch_shape,
    channels,
    with_v,
    seq_v,
    dtype,
    as_variable,
    device,
    compile_graph,
    call,
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    x = ivy.astype(
        ivy.linspace(
            ivy.zeros(batch_shape),
            ivy.ones(batch_shape),
            channels,
            axis=1,
            dtype=dtype,
            device=device,
        ),
        dtype=dtype,
    )
    if as_variable:
        x = ivy.variable(x)
    if with_v:
        np.random.seed(0)
        wlim = (6 / (channels + channels)) ** 0.5
        v = ivy.Container(
            {
                "submodules": {
                    "v0": {
                        "w": ivy.variable(
                            ivy.asarray(
                                np.random.uniform(-wlim, wlim, (channels, channels)),
                                dtype=dtype,
                                device=device,
                            )
                        ),
                        "b": ivy.variable(
                            ivy.zeros([channels], device=device, dtype=dtype)
                        ),
                    },
                    "v1": {
                        "w": ivy.variable(
                            ivy.asarray(
                                np.random.uniform(-wlim, wlim, (channels, channels)),
                                dtype=dtype,
                                device=device,
                            )
                        ),
                        "b": ivy.variable(
                            ivy.zeros([channels], device=device, dtype=dtype)
                        ),
                    },
                    "v2": {
                        "w": ivy.variable(
                            ivy.asarray(
                                np.random.uniform(-wlim, wlim, (channels, channels)),
                                dtype=dtype,
                                device=device,
                            )
                        ),
                        "b": ivy.variable(
                            ivy.zeros([channels], device=device, dtype=dtype)
                        ),
                    },
                }
            }
        )
    else:
        v = None
    if seq_v:
        seq = ivy.Sequential(
            ivy.Linear(channels, channels, device=device),
            ivy.Linear(channels, channels, device=device),
            ivy.Linear(channels, channels, device=device),
            v=v if with_v else None,
            device=device,
        )
    else:
        seq = ivy.Sequential(
            ivy.Linear(
                channels,
                channels,
                device=device,
                v=v["submodules"]["v0"] if with_v else None,
            ),
            ivy.Linear(
                channels,
                channels,
                device=device,
                v=v["submodules"]["v1"] if with_v else None,
            ),
            ivy.Linear(
                channels,
                channels,
                device=device,
                v=v["submodules"]["v2"] if with_v else None,
            ),
            device=device,
        )

    def loss_fn(x_, v_):
        out = seq(x_, v=v_)
        return ivy.mean(out)

    def train_step(x_, v_):
        lss, grds = ivy.execute_with_gradients(lambda _v_: loss_fn(x_, _v_), v_)
        v_ = ivy.gradient_descent_update(seq.v, grds, 1e-3)
        return lss, grds, v_

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads, seq.v = train_step(x, seq.v)
        assert ivy.to_scalar(loss) < ivy.to_scalar(loss_tm1)
        loss_tm1 = loss

    # type test
    assert ivy.is_array(loss)
    assert isinstance(grads, ivy.Container)
    # cardinality test
    if call is helpers.mx_call:
        # mxnet slicing cannot reduce dimension to zero
        assert loss.shape == (1,)
    else:
        assert loss.shape == ()
    # value test
    assert (abs(grads).max() > 0).all_true()
