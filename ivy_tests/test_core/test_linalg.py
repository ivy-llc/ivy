"""
Collection of tests for templated linear algebra functions
"""

# global
import numpy as np
from functools import reduce as _reduce
from operator import mul as _mul

# local
import ivy.core.linalg as ivy_linalg
import ivy.core.general as ivy_gen
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
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support svd
            continue
        pred = call(ivy_linalg.svd, ivy_gen.array([[[1., 0.], [0., 1.]]], f=lib))
        true = np.linalg.svd(np.array([[[1., 0.], [0., 1.]]]))
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1
        pred = call(ivy_linalg.svd, ivy_gen.array([[[[1., 0.], [0., 1.]]]], f=lib))
        true = np.linalg.svd(np.array([[[[1., 0.], [0., 1.]]]]))
        assert _reduce(_mul, [np.array_equal(pred_, true_) for pred_, true_ in zip(pred, true)], 1) == 1


def test_norm():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_linalg.norm, ivy_gen.array([[1., 0.], [0., 1.]], f=lib), 1),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, -1))
        assert np.array_equal(call(ivy_linalg.norm, ivy_gen.array([[1., 0.], [0., 1.]], f=lib), 1, 1),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1))
        assert np.array_equal(call(ivy_linalg.norm, ivy_gen.array([[1., 0.], [0., 1.]], f=lib), 1, 1, True),
                              np.linalg.norm(np.array([[1., 0.], [0., 1.]]), 1, 1, True))
        assert np.array_equal(call(ivy_linalg.norm, ivy_gen.array([[[1., 0.], [0., 1.]]], f=lib), 2),
                              np.linalg.norm(np.array([[[1., 0.], [0., 1.]]]), 2, -1))


def test_inv():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_linalg.inv, ivy_gen.array([[1., 0.], [0., 1.]], f=lib)),
                              np.linalg.inv(np.array([[1., 0.], [0., 1.]])))
        assert np.array_equal(call(ivy_linalg.inv, ivy_gen.array([[[1., 0.], [0., 1.]]], f=lib)),
                              np.linalg.inv(np.array([[[1., 0.], [0., 1.]]])))


def test_pinv():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support pinv
            continue
        assert np.allclose(call(ivy_linalg.pinv, ivy_gen.array([[1., 0.], [0., 1.], [1., 0.]], f=lib)),
                              np.linalg.pinv(np.array([[1., 0.], [0., 1.], [1., 0.]])), atol=1e-6)
        assert np.allclose(call(ivy_linalg.pinv, ivy_gen.array([[[1., 0.], [0., 1.], [1., 0.]]], f=lib)),
                              np.linalg.pinv(np.array([[[1., 0.], [0., 1.], [1., 0.]]])), atol=1e-6)


def test_vector_to_skew_symmetric_matrix():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_linalg.vector_to_skew_symmetric_matrix, td.vectors),
                           td.skew_symmetric_matrices, atol=1e-6)
        assert np.allclose(call(ivy_linalg.vector_to_skew_symmetric_matrix, td.vectors[0]),
                           td.skew_symmetric_matrices[0], atol=1e-6)
