"""
Collection of tests for templated image-related functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# stack_images
@pytest.mark.parametrize(
    "shp_n_num_n_ar_n_newshp", [((1, 2, 3), 4, (2, 1), (2, 4, 3)),
                                ((8, 8, 3), 9, (1, 1), (24, 24, 3)),
                                ((3, 16, 12, 4), 10, (2, 5), (3, 80, 36, 4)),
                                ((5, 20, 9, 5), 10, (5, 2), (5, 40, 72, 5))])
def test_stack_images(shp_n_num_n_ar_n_newshp, dev_str, call):
    # smoke test
    shape, num, ar, new_shape = shp_n_num_n_ar_n_newshp
    xs = [ivy.ones(shape)] * num
    ret = ivy.stack_images(xs, ar)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == new_shape
    # compilation test
    helpers.assert_compilable(ivy.stack_images)


# bilinear_resample
@pytest.mark.parametrize(
    "x_n_warp", [([[[[0.], [1.]], [[2.], [3.]]]], [[[[0., 1.], [0.5, 0.5]], [[0.5, 1.], [1., 0.5]]]]),
                 ([[[[[0.], [1.]], [[2.], [3.]]]]], [[[[[0., 1.], [0.5, 0.5]], [[0.5, 1.], [1., 0.5]]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_bilinear_resample(x_n_warp, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, warp = x_n_warp
    x = tensor_fn(x, dtype_str, dev_str)
    warp = tensor_fn(warp, dtype_str, dev_str)
    ret = ivy.bilinear_resample(x, warp)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.bilinear_resample, x, warp),
                       ivy.numpy.bilinear_resample(ivy.to_numpy(x), ivy.to_numpy(warp)))
    # compilation test
    if call in [helpers.torch_call]:
        # torch scripting does not support builtins
        return
    helpers.assert_compilable(ivy.bilinear_resample)


# gradient_image
@pytest.mark.parametrize(
    "x_n_dy_n_dx", [([[[[0.], [1.], [2.]], [[5.], [4.], [3.]], [[6.], [8.], [7.]]]],
                     [[[[5.], [3.], [1.]], [[1.], [4.], [4.]], [[0.], [0.], [0.]]]],
                     [[[[1.], [1.], [0.]], [[-1.], [-1.], [0.]], [[2.], [-1.], [0.]]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gradient_image(x_n_dy_n_dx, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, dy_true, dx_true = x_n_dy_n_dx
    x = tensor_fn(x, dtype_str, dev_str)
    dy, dx = ivy.gradient_image(x)
    # type test
    assert ivy.is_array(dy)
    assert ivy.is_array(dx)
    # cardinality test
    assert dy.shape == x.shape
    assert dx.shape == x.shape
    # value test
    dy_np, dx_np = call(ivy.gradient_image, x)
    dy_true = ivy.numpy.array(dy_true, dtype_str, dev_str)
    dx_true = ivy.numpy.array(dx_true, dtype_str, dev_str)
    assert np.allclose(dy_np, dy_true)
    assert np.allclose(dx_np, dx_true)
    # compilation test
    if call in [helpers.torch_call]:
        # torch device cannot be assigned value of string while scripting
        return
    helpers.assert_compilable(ivy.gradient_image)
