"""
Collection of tests for templated image-related functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_stack_images():
    for f, call in helpers.f_n_calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # only for visualization, and so only supported in eager mode
            continue
        assert call(ivy.stack_images, [ivy.ones((1, 2, 3), f=f)] * 4, (2, 1)).shape == (2, 4, 3)
        assert call(ivy.stack_images, [ivy.ones((8, 8, 3), f=f)] * 9, (1, 1)).shape == (24, 24, 3)
        assert call(ivy.stack_images, [ivy.ones((3, 16, 12, 4), f=f)] * 10, (2, 5)).shape == (3, 80, 36, 4)
        assert call(ivy.stack_images, [ivy.ones((5, 20, 9, 5), f=f)] * 10, (5, 2)).shape == (5, 40, 72, 5)
        helpers.assert_compilable('stack_images', f)


# noinspection PyTypeChecker
def test_bilinear_resample():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # Mxnet symbolic has no BilinearSampler class
            continue
        assert np.array_equal(call(ivy.bilinear_resample, ivy.array([[[[0.], [1.]],
                                                                      [[2.], [3.]]]], f=f),
                                   ivy.array([[[[0., 1.], [0.5, 0.5]],
                                               [[0.5, 1.], [1., 0.5]]]], f=f), [1], [2, 2]),

                              ivy.bilinear_resample(np.array([[[[0.], [1.]],
                                                               [[2.], [3.]]]]),
                                                    np.array([[[[0., 1.], [0.5, 0.5]],
                                                                  [[0.5, 1.], [1., 0.5]]]])))

        assert np.array_equal(call(ivy.bilinear_resample, ivy.array([[[[[0.], [1.]],
                                                                       [[2.], [3.]]]]], f=f),
                                   ivy.array([[[[[0., 1.], [0.5, 0.5]],
                                                [[0.5, 1.], [1., 0.5]]]]], f=f), [1, 1], [2, 2]),
                              ivy.bilinear_resample(np.array([[[[[0.], [1.]],
                                                                [[2.], [3.]]]]]),
                                                    np.array([[[[[0., 1.], [0.5, 0.5]],
                                                                   [[0.5, 1.], [1., 0.5]]]]])))
        if call in [helpers.torch_call]:
            # torch scripting does not support builtins
            continue
        helpers.assert_compilable('bilinear_resample', f)


def test_gradient_image():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # Mxnet symbolic does not fully support array slicing
            continue
        dy, dx = call(ivy.gradient_image, ivy.array([[[[0.], [1.], [2.]],
                                                      [[5.], [4.], [3.]],
                                                      [[6.], [8.], [7.]]]], f=f), [1], [3, 3])
        assert np.array_equal(dy, np.array([[[[5.], [3.], [1.]],
                                             [[1.], [4.], [4.]],
                                             [[0.], [0.], [0.]]]]))
        assert np.array_equal(dx, np.array([[[[1.], [1.], [0.]],
                                             [[-1.], [-1.], [0.]],
                                             [[2.], [-1.], [0.]]]]))
        helpers.assert_compilable('gradient_image', f)
