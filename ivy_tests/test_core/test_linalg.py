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


def test_svd(dev_str, call):
    pred = call(ivy.svd, ivy.tensor([[[1., 0.], [0., 1.]]]))
    true = np.linalg.svd(np.array([[[1., 0.], [0., 1.]]]))
    assert reduce(mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
    pred = call(ivy.svd, ivy.tensor([[[[1., 0.], [0., 1.]]]]))
    true = np.linalg.svd(np.array([[[[1., 0.], [0., 1.]]]]))
    assert reduce(mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
    helpers.assert_compilable(ivy.svd)


def test_norm(dev_str, call):
    assert np.array_equal(call(ivy.norm, ivy.tensor([[1., 0.], [0., 1.]]), 1, -1),
                          np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, -1))
    assert np.array_equal(call(ivy.norm, ivy.tensor([[1., 0.], [0., 1.]]), 1, 1),
                          np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1))
    assert np.array_equal(call(ivy.norm, ivy.tensor([[1., 0.], [0., 1.]]), 1, 1, True),
                          np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1, True))
    assert np.array_equal(call(ivy.norm, ivy.tensor([[[1., 0.], [0., 1.]]]), 2, -1),
                          np.linalg.norm(np.array([[[1., 0.], [0., 1.]]]), 2, -1))
    helpers.assert_compilable(ivy.norm)


def test_inv(dev_str, call):
    assert np.array_equal(call(ivy.inv, ivy.tensor([[1., 0.], [0., 1.]])),
                          np.linalg.inv(np.array([[1., 0.], [0., 1.]])))
    assert np.array_equal(call(ivy.inv, ivy.tensor([[[1., 0.], [0., 1.]]])),
                          np.linalg.inv(np.array([[[1., 0.], [0., 1.]]])))
    helpers.assert_compilable(ivy.inv)


def test_pinv(dev_str, call):
    assert np.allclose(call(ivy.pinv, ivy.tensor([[1., 0.], [0., 1.], [1., 0.]])),
                       np.linalg.pinv(np.array([[1., 0.], [0., 1.], [1., 0.]])), atol=1e-6)
    assert np.allclose(call(ivy.pinv, ivy.tensor([[[1., 0.], [0., 1.], [1., 0.]]])),
                       np.linalg.pinv(np.array([[[1., 0.], [0., 1.], [1., 0.]]])), atol=1e-6)
    helpers.assert_compilable(ivy.pinv)


def test_vector_to_skew_symmetric_matrix(dev_str, call):
    assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, td.vectors),
                       td.skew_symmetric_matrices, atol=1e-6)
    assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, td.vectors[0]),
                       td.skew_symmetric_matrices[0], atol=1e-6)
    helpers.assert_compilable(ivy.vector_to_skew_symmetric_matrix)
