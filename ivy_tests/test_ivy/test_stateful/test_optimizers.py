"""Collection of tests for Ivy optimizers."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy
from ivy.container import Container
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# sgd
@given(
    bs_ic_oc_target=st.sampled_from(
        [
            (
                [1, 2],
                4,
                5,
                [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]],
            ),
        ]
    ),
    with_v=st.booleans(),
    inplace=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_sgd_optimizer(
    bs_ic_oc_target, with_v, inplace, dtype, device, compile_graph, call
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([output_channels]))
        v = Container({"w": w, "b": b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, device=device, v=v)

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.mean(out)

    # optimizer
    optimizer = ivy.SGD(inplace=ivy.inplace_variables_supported() if inplace else False)

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
        assert loss < loss_tm1
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
    assert ivy.max(ivy.abs(grads.b)) > 0
    assert ivy.max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# lars
@given(
    bs_ic_oc_target=st.sampled_from(
        [
            (
                [1, 2],
                4,
                5,
                [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]],
            ),
        ]
    ),
    with_v=st.booleans(),
    inplace=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_lars_optimizer(
    bs_ic_oc_target, with_v, inplace, dtype, device, compile_graph, call
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        dtype="float32",
    )
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([output_channels]))
        v = Container({"w": w, "b": b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, device=device, v=v)

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.mean(out)

    # optimizer
    optimizer = ivy.LARS(
        inplace=ivy.inplace_variables_supported() if inplace else False
    )

    # train
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
        assert loss < loss_tm1
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
    assert ivy.max(ivy.abs(grads.b)) > 0
    assert ivy.max(ivy.abs(grads.w)) > 0
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return


# adam
@given(
    bs_ic_oc_target=st.sampled_from(
        [
            (
                [1, 2],
                4,
                5,
                [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]],
            ),
        ]
    ),
    with_v=st.booleans(),
    inplace=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_adam_optimizer(
    bs_ic_oc_target, with_v, inplace, dtype, device, compile_graph, call
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        dtype="float32",
    )
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([output_channels]))
        v = Container({"w": w, "b": b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, device=device, v=v)

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.mean(out)

    # optimizer
    optimizer = ivy.Adam(
        device=device, inplace=ivy.inplace_variables_supported() if inplace else False
    )

    # train
    loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
    linear_layer.v = optimizer.step(linear_layer.v, grads)
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
        assert loss < loss_tm1
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
    assert ivy.max(ivy.abs(grads.b)) > 0
    assert ivy.max(ivy.abs(grads.w)) > 0


# lamb
@given(
    bs_ic_oc_target=st.sampled_from(
        [
            (
                [1, 2],
                4,
                5,
                [[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]],
            ),
        ]
    ),
    with_v=st.booleans(),
    inplace=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_lamb_optimizer(
    bs_ic_oc_target, with_v, inplace, dtype, device, compile_graph, call
):
    # smoke test
    if call is helpers.np_call:
        # NumPy does not support gradients
        return
    batch_shape, input_channels, output_channels, target = bs_ic_oc_target
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        dtype="float32",
    )
    if with_v:
        np.random.seed(0)
        wlim = (6 / (output_channels + input_channels)) ** 0.5
        w = ivy.variable(
            ivy.array(
                np.random.uniform(-wlim, wlim, (output_channels, input_channels)),
                dtype="float32",
                device=device,
            )
        )
        b = ivy.variable(ivy.zeros([output_channels]))
        v = Container({"w": w, "b": b})
    else:
        v = None
    linear_layer = ivy.Linear(input_channels, output_channels, device=device, v=v)

    def loss_fn(v_):
        out = linear_layer(x, v=v_)
        return ivy.mean(out)

    # optimizer
    optimizer = ivy.LAMB(
        device=device, inplace=ivy.inplace_variables_supported() if inplace else False
    )

    # train
    loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
    linear_layer.v = optimizer.step(linear_layer.v, grads)
    loss_tm1 = 1e12
    loss = None
    grads = None
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, linear_layer.v)
        linear_layer.v = optimizer.step(linear_layer.v, grads)
        assert loss < loss_tm1
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
    assert ivy.max(ivy.abs(grads.b)) > 0
    assert ivy.max(ivy.abs(grads.w)) > 0
