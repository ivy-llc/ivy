"""
Collection of tests for unified linear algebra functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# svd
# @pytest.mark.parametrize(
#     "x", [[[[1., 0.], [0., 1.]]], [[[[1., 0.], [0., 1.]]]]])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_svd(x, dtype, tensor_fn, dev, call):
#     if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev:
#         # tf.linalg.svd segfaults when CUDA is installed, but array is on CPU
#         pytest.skip()
#     # smoke test
#     x = tensor_fn(x, dtype, dev)
#     u, s, vh = ivy.svd(x)
#     # type test
#     assert ivy.is_array(u)
#     assert ivy.is_array(s)
#     assert ivy.is_array(vh)
#     # cardinality test
#     assert u.shape == x.shape
#     assert s.shape == x.shape[:-1]
#     assert vh.shape == x.shape
#     # value test
#     pred_u, pred_s, pred_vh = call(ivy.svd, x)
#     true_u, true_s, true_vh = ivy.functional.backends.numpy.svd(ivy.to_numpy(x))
#     assert np.allclose(pred_u, true_u)
#     assert np.allclose(pred_s, true_s)
#     assert np.allclose(pred_vh, true_vh)


# vector_norm
# @pytest.mark.parametrize(
#     "x_n_p_n_ax_n_kd_n_tn", [([[1., 2.], [3., 4.]], 2, -1, None, [2.236068, 5.0]),
#                              ([[1., 2.], [3., 4.]], 3, None, False, 4.641588),
#                              ([[1., 2.], [3., 4.]], -float('inf'), None, False, 1.),
#                              ([[1., 2.], [3., 4.]], 0, None, False, 4.),
#                              ([[1., 2.], [3., 4.]], float('inf'), None, False, 4.),
#                              ([[1., 2.], [3., 4.]], 0.5, 0, True, [[7.464102, 11.656854]]),
#                              ([[[1., 2.], [3., 4.]]], 1, None, None, 10.)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_vector_norm(x_n_p_n_ax_n_kd_n_tn, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, p, ax, kd, true_norm = x_n_p_n_ax_n_kd_n_tn
#     x = tensor_fn(x, dtype, dev)
#     kwargs = {k: v for k, v in zip(['x', 'p', 'axis', 'keepdims'], [x, p, ax, kd]) if v is not None}
#     ret = ivy.vector_norm(**kwargs)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     if kd:
#         expected_shape = [1 if i == ax else item for i, item in enumerate(x.shape)]
#     elif ax is None:
#         expected_shape = [1]
#     else:
#         expected_shape = list(x.shape)
#         expected_shape.pop(ax)
#     assert ret.shape == tuple(expected_shape)
#     # value test
#     kwargs.pop('x', None)
#     assert np.allclose(call(ivy.vector_norm, x, **kwargs), np.array(true_norm))


# matrix_norm
@pytest.mark.parametrize(
    "x_n_p_n_ax_n_kd", [([[[1., 2.], [3., 4.]]], 2, (-2, -1), None),
                        ([[1., 2.], [3., 4.]], -2, None, False),
                        ([[1., 2.], [3., 4.]], -float('inf'), None, False),
                        ([[1., 2.], [3., 4.]], float('inf'), None, False),
                        ([[[1.], [2.]], [[3.], [4.]]], 1, (0, 1), True),
                        ([[[1., 2.], [3., 4.]]], -1, None, None)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_matrix_norm(x_n_p_n_ax_n_kd, dtype, tensor_fn, dev, call):
    # smoke test
    x_raw, p, ax, kd = x_n_p_n_ax_n_kd
    if p == -2 and call in [helpers.tf_call, helpers.tf_graph_call]:
        # tensorflow does not support these p value of -2
        pytest.skip()
    if call is helpers.mx_call:
        # MXNet does not support matrix norms
        pytest.skip()
    x = tensor_fn(x_raw, dtype, dev)
    kwargs = {k: v for k, v in zip(['x', 'p', 'axes', 'keepdims'], [x, p, ax, kd]) if v is not None}
    ret = ivy.matrix_norm(**kwargs)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if kd:
        expected_shape = [1] * len(x.shape)
    elif ax is None:
        if len(x.shape) > 2:
            expected_shape = [1] * (len(x.shape) - 2)
        else:
            expected_shape = [1]
    else:
        expected_shape = [1 for i, item in enumerate(x.shape) if i not in [a % len(x.shape) for a in ax]]
    assert ret.shape == tuple(expected_shape)
    # value test
    kwargs.pop('x', None)
    pred = call(ivy.matrix_norm, x, **kwargs)
    if 'p' in kwargs:
        kwargs['ord'] = kwargs['p']
        del kwargs['p']
    if 'axes' in kwargs:
        kwargs['axis'] = kwargs['axes']
        del kwargs['axes']
    else:
        kwargs['axis'] = (-2, -1)
    assert np.allclose(pred, np.linalg.norm(np.array(x_raw), **kwargs))


# pinv
@pytest.mark.parametrize(
    "x", [[[1., 0.], [0., 1.], [1., 0.]], [[[1., 0.], [0., 1.], [1., 0.]]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_pinv(x, dtype, tensor_fn, dev, call):
    if call in [helpers.tf_call, helpers.tf_graph_call] and 'cpu' in dev:
        # tf.linalg.pinv segfaults when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.pinv(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape[:-2] + (x.shape[-1], x.shape[-2])
    # value test
    assert np.allclose(call(ivy.pinv, x), ivy.functional.backends.numpy.pinv(ivy.to_numpy(x)))


# vector_to_skew_symmetric_matrix
@pytest.mark.parametrize(
    "x", [[[[1., 2., 3.]], [[4., 5., 6.]], [[1., 2., 3.]], [[4., 5., 6.]], [[1., 2., 3.]]], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_vector_to_skew_symmetric_matrix(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.vector_to_skew_symmetric_matrix(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape + (x.shape[-1],)
    # value test
    assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, x),
                       ivy.functional.backends.numpy.vector_to_skew_symmetric_matrix(ivy.to_numpy(x)))


# cholesky
@pytest.mark.parametrize(
    "x", [[[1.0, -1.0, 2.0], [-1.0, 5.0, -4.0], [2.0, -4.0, 6.0]], [[[1.0, -1.0, 2.0], [-1.0, 5.0, -4.0], [2.0, -4.0, 6.0]]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cholesky(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.cholesky(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cholesky, x), ivy.functional.backends.numpy.cholesky(ivy.to_numpy(x)))

# QR
# @pytest.mark.parametrize(
#     "data", [ ([[-1.0, 1.0, 3], [4, -5, 8], [9, 2, -3]], 3),
#            ([[[-1.0, 1.0, 3], [4, -5, 8], [9, 2, -3], [12, -2, 8]]], 3),
#      ])
# @pytest.mark.parametrize(
#     "mode", ['reduced', 'complete'])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_qr(data, mode, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, rank = data
#     x = tensor_fn(x, dtype, dev)
#     Q, R = ivy.qr(x, mode)
#     # type test
#     assert ivy.is_array(Q)
#     assert ivy.is_array(R)
#     # cardinality test
#     if mode == "reduced":
#         assert Q.shape[-2] == x.shape[-2]
#         assert Q.shape[-1] == rank
#     if mode == "complete":
#         assert Q.shape[-2] == x.shape[-2]
#         assert Q.shape[-1] == x.shape[-2]
#     # value test
#     assert np.allclose(call(ivy.qr, x, mode)[0], ivy.functional.backends.numpy.qr(ivy.to_numpy(x), mode)[0])