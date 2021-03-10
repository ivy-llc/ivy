"""
Collection of tests for templated linear algebra functions
"""

# global
import numpy as np
from operator import mul
from functools import reduce

# local
import ivy
import ivy_tests.helpers as helpers


class LinAlgTestData:

    def __init__(self):

        self.vectors = np.array([[[1., 2., 3.]],
                                 [[4., 5., 6.]],
                                 [[1., 2., 3.]],
                                 [[4., 5., 6.]],
                                 [[1., 2., 3.]]])

        self.skew_symmetric_matrices = np.array([[[[0., -3., 2.],
                                                   [3., 0., -1.],
                                                   [-2., 1., 0.]]],

                                                 [[[0., -6., 5.],
                                                   [6., 0., -4.],
                                                   [-5., 4., 0.]]],

                                                 [[[0., -3., 2.],
                                                   [3., 0., -1.],
                                                   [-2., 1., 0.]]],

                                                 [[[0., -6., 5.],
                                                   [6., 0., -4.],
                                                   [-5., 4., 0.]]],

                                                 [[[0., -3., 2.],
                                                   [3., 0., -1.],
                                                   [-2., 1., 0.]]]])


td = LinAlgTestData()


def test_svd():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support svd
            continue
        pred = call(ivy.svd, ivy.array([[[1., 0.], [0., 1.]]], f=f))
        true = np.linalg.svd(np.array([[[1., 0.], [0., 1.]]]))
        assert reduce(mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        pred = call(ivy.svd, ivy.array([[[[1., 0.], [0., 1.]]]], f=f))
        true = np.linalg.svd(np.array([[[[1., 0.], [0., 1.]]]]))
        assert reduce(mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        helpers.assert_compilable('svd', f)


def test_norm():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.norm, ivy.array([[1., 0.], [0., 1.]], f=f), 1),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, -1))
        assert np.array_equal(call(ivy.norm, ivy.array([[1., 0.], [0., 1.]], f=f), 1, 1),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1))
        assert np.array_equal(call(ivy.norm, ivy.array([[1., 0.], [0., 1.]], f=f), 1, 1, True),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1, True))
        assert np.array_equal(call(ivy.norm, ivy.array([[[1., 0.], [0., 1.]]], f=f), 2),
                              np.linalg.norm(np.array([[[1., 0.], [0., 1.]]]), 2, -1))
        helpers.assert_compilable('norm', f)


def test_inv():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.inv, ivy.array([[1., 0.], [0., 1.]], f=f)),
                              np.linalg.inv(np.array([[1., 0.], [0., 1.]])))
        assert np.array_equal(call(ivy.inv, ivy.array([[[1., 0.], [0., 1.]]], f=f)),
                              np.linalg.inv(np.array([[[1., 0.], [0., 1.]]])))
        helpers.assert_compilable('inv', f)


def test_pinv():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support pinv
            continue
        assert np.allclose(call(ivy.pinv, ivy.array([[1., 0.], [0., 1.], [1., 0.]], f=f)),
                           np.linalg.pinv(np.array([[1., 0.], [0., 1.], [1., 0.]])), atol=1e-6)
        assert np.allclose(call(ivy.pinv, ivy.array([[[1., 0.], [0., 1.], [1., 0.]]], f=f)),
                           np.linalg.pinv(np.array([[[1., 0.], [0., 1.], [1., 0.]]])), atol=1e-6)
        helpers.assert_compilable('pinv', f)


def test_vector_to_skew_symmetric_matrix():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, td.vectors),
                           td.skew_symmetric_matrices, atol=1e-6)
        assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, td.vectors[0]),
                           td.skew_symmetric_matrices[0], atol=1e-6)
        helpers.assert_compilable('vector_to_skew_symmetric_matrix', f)
