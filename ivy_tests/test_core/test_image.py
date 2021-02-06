"""
Collection of tests for templated image-related functions
"""

# global
import numpy as np

# local
import ivy.core.image as ivy_im
import ivy.core.general as ivy_gen
import ivy_tests.helpers as helpers


def test_stack_images():
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # only for visualization, and so only supported in eager mode
            continue
        assert call(ivy_im.stack_images, [ivy_gen.ones((1, 2, 3), f=lib)] * 4, (2, 1)).shape == (2, 4, 3)
        assert call(ivy_im.stack_images, [ivy_gen.ones((8, 8, 3), f=lib)] * 9, (1, 1)).shape == (24, 24, 3)
        assert call(ivy_im.stack_images, [ivy_gen.ones((3, 16, 12, 4), f=lib)] * 10, (2, 5)).shape == (3, 80, 36, 4)
        assert call(ivy_im.stack_images, [ivy_gen.ones((5, 20, 9, 5), f=lib)] * 10, (5, 2)).shape == (5, 40, 72, 5)


# noinspection PyTypeChecker
def test_bilinear_resample():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # Mxnet symbolic has no BilinearSampler class
            continue
        assert np.array_equal(call(ivy_im.bilinear_resample, ivy_gen.array([[[[0.], [1.]],
                                                                             [[2.], [3.]]]], f=lib),
                                   ivy_gen.array([[[[0., 1.], [0.5, 0.5]],
                                                   [[0.5, 1.], [1., 0.5]]]], f=lib), [1], [2, 2]),

                              ivy_im.bilinear_resample(np.array([[[[0.], [1.]],
                                                                  [[2.], [3.]]]]),
                                                       np.array([[[[0., 1.], [0.5, 0.5]],
                                                                  [[0.5, 1.], [1., 0.5]]]])))

        assert np.array_equal(call(ivy_im.bilinear_resample, ivy_gen.array([[[[[0.], [1.]],
                                                                              [[2.], [3.]]]]], f=lib),
                                   ivy_gen.array([[[[[0., 1.], [0.5, 0.5]],
                                                    [[0.5, 1.], [1., 0.5]]]]], f=lib), [1, 1], [2, 2]),
                              ivy_im.bilinear_resample(np.array([[[[[0.], [1.]],
                                                                   [[2.], [3.]]]]]),
                                                       np.array([[[[[0., 1.], [0.5, 0.5]],
                                                                   [[0.5, 1.], [1., 0.5]]]]])))


def test_gradient_image():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # Mxnet symbolic does not fully support array slicing
            continue
        dy, dx = call(ivy_im.gradient_image, ivy_gen.array([[[[0.], [1.], [2.]],
                                                             [[5.], [4.], [3.]],
                                                             [[6.], [8.], [7.]]]], f=lib), [1], [3, 3])
        assert np.array_equal(dy, np.array([[[[5.], [3.], [1.]],
                                             [[1.], [4.], [4.]],
                                             [[0.], [0.], [0.]]]]))
        assert np.array_equal(dx, np.array([[[[1.], [1.], [0.]],
                                             [[-1.], [-1.], [0.]],
                                             [[2.], [-1.], [0.]]]]))
