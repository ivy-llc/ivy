"""
Collection of tests for templated linear algebra functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# svd
@pytest.mark.parametrize(
    "x", [[[[1., 0.], [0., 1.]]], [[[[1., 0.], [0., 1.]]]]]
)
@pytest.mark.parametrize("dtype_str", ['float32'])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_svd(x, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf.linalg.svd segfaults when CUDA is installed, but array is on CPU
        pytest.skip()

    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    u, s, vh = ivy.svd(x)

    # type test
    assert ivy.is_array(u)
    assert ivy.is_array(s)
    assert ivy.is_array(vh)

    # cardinality test
    assert u.shape == x.shape
    assert s.shape == x.shape[:-1]
    assert vh.shape == x.shape

    # value test
    pred_u, pred_s, pred_vh = call(ivy.svd, x)
    true_u, true_s, true_vh = ivy.numpy.svd(ivy.to_numpy(x))
    assert np.allclose(pred_u, true_u)
    assert np.allclose(pred_s, true_s)
    assert np.allclose(pred_vh, true_vh)

    # compilation test
    helpers.assert_compilable(ivy.svd)


# norm
@pytest.mark.parametrize(
    "x_n_ord_n_ax_n_kd",
    [
        ([[1., 0.], [0., 1.]], 1, -1, None),
        ([[1., 0.], [0., 1.]], 1, 1, None),
        ([[1., 0.], [0., 1.]], 1, 1, True),
        ([[[1., 0.], [0., 1.]]], 2, -1, None)
    ]
)
@pytest.mark.parametrize("dtype_str", ['float32'])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_norm(x_n_ord_n_ax_n_kd, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, order, ax, kd = x_n_ord_n_ax_n_kd
    x = tensor_fn(x, dtype_str, dev_str)
    kwargs = dict(
        [
            (k, v) for k, v in zip(
                ['x', 'ord', 'axis', 'keepdims'], [x, order, ax, kd]
            ) if v is not None
        ]
    )

    ret = ivy.norm(**kwargs)

    # type test
    assert ivy.is_array(ret)

    # cardinality test
    if kd:
        expected_shape = [
            1 if i == ax else item for i, item in enumerate(x.shape)
        ]
    else:
        expected_shape = list(x.shape)
        expected_shape.pop(ax)

    assert ret.shape == tuple(expected_shape)

    # value test
    kwargs.pop('x', None)
    assert np.allclose(call(ivy.norm, x, **kwargs), ivy.numpy.norm(ivy.to_numpy(x), **kwargs))

    # compilation test
    helpers.assert_compilable(ivy.norm)


# inv
@pytest.mark.parametrize("x", [[[1., 0.], [0., 1.]], [[[1., 0.], [0., 1.]]]])
@pytest.mark.parametrize("dtype_str", ['float32'])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_inv(x, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf.linalg.inv segfaults when CUDA is installed, but array is on CPU
        pytest.skip()

    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.inv(x)

    # type test
    assert ivy.is_array(ret)

    # cardinality test
    assert ret.shape == x.shape

    # value test
    assert np.allclose(call(ivy.inv, x), ivy.numpy.inv(ivy.to_numpy(x)))

    # compilation test
    helpers.assert_compilable(ivy.inv)


# pinv
@pytest.mark.parametrize(
    "x", [[[1., 0.], [0., 1.], [1., 0.]], [[[1., 0.], [0., 1.], [1., 0.]]]]
)
@pytest.mark.parametrize("dtype_str", ['float32'])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_pinv(x, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev_str:
        # tf.linalg.pinv segfaults when CUDA is installed, but array is on CPU
        pytest.skip()

    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.pinv(x)

    # type test
    assert ivy.is_array(ret)

    # cardinality test
    assert ret.shape == x.shape[:-2] + (x.shape[-1], x.shape[-2])

    # value test
    assert np.allclose(call(ivy.pinv, x), ivy.numpy.pinv(ivy.to_numpy(x)))

    # compilation test
    helpers.assert_compilable(ivy.pinv)


# vector_to_skew_symmetric_matrix
@pytest.mark.parametrize(
    "x",
    [
        [
            [[1., 2., 3.]],
            [[4., 5., 6.]],
            [[1., 2., 3.]],
            [[4., 5., 6.]],
            [[1., 2., 3.]]
        ],
        [[1., 2., 3.]]
    ]
)
@pytest.mark.parametrize("dtype_str", ['float32'])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_vector_to_skew_symmetric_matrix(
        x, dtype_str, tensor_fn, dev_str, call
):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.vector_to_skew_symmetric_matrix(x)

    # type test
    assert ivy.is_array(ret)

    # cardinality test
    assert ret.shape == x.shape + (x.shape[-1],)

    # value test
    assert np.allclose(
        call(ivy.vector_to_skew_symmetric_matrix, x),
        ivy.numpy.vector_to_skew_symmetric_matrix(ivy.to_numpy(x))
    )

    # compilation test
    helpers.assert_compilable(ivy.vector_to_skew_symmetric_matrix)
