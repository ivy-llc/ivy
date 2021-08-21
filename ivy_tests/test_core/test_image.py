"""
Collection of tests for templated image-related functions
"""

# global
import pytest
import numpy as np
from operator import mul
# noinspection PyProtectedMember
from functools import reduce

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
    "x_n_warp", [([[[[0.], [1.]], [[2.], [3.]]]], [[[0., 1.], [0.25, 0.25], [0.5, 0.5], [0.5, 1.], [1., 0.5]]]),
                 ([[[[0.], [1.]], [[2.], [3.]]]], [[[0., 1.], [0.5, 0.5], [0.5, 1.], [1., 0.5]]]),
                 ([[[[[0.], [1.]], [[2.], [3.]]]]], [[[[0., 1.], [0.5, 0.5], [0.5, 1.], [1., 0.5]]]])])
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
    assert ret.shape == warp.shape[:-1] + x.shape[-1:]
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


# float_img_to_uint8_img
@pytest.mark.parametrize(
    "fi_tui", [([[0., 1.], [2., 3.]],
               [[[0, 0, 0, 0], [0, 0, 128, 63]], [[0, 0, 0, 64], [0, 0, 64, 64]]])])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_float_img_to_uint8_img(fi_tui, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.tf_graph_call:
        # tensorflow tensors cannot be cast to numpy arrays in graph mode
        pytest.skip()
    float_img, true_uint8_img = fi_tui
    float_img = tensor_fn(float_img, 'float32', dev_str)
    true_uint8_img = np.array(true_uint8_img)
    uint8_img = ivy.float_img_to_uint8_img(float_img)
    # type test
    assert ivy.is_array(float_img)
    # cardinality test
    assert uint8_img.shape == true_uint8_img.shape
    # value test
    uint8_img_np = call(ivy.float_img_to_uint8_img, float_img)
    assert np.allclose(uint8_img_np, true_uint8_img)
    # compilation test
    if call in [helpers.torch_call]:
        # torch device cannot be assigned value of string while scripting
        return
    helpers.assert_compilable(ivy.float_img_to_uint8_img)


# uint8_img_to_float_img
@pytest.mark.parametrize(
    "ui_tfi", [([[[0, 0, 0, 0], [0, 0, 128, 63]], [[0, 0, 0, 64], [0, 0, 64, 64]]],
                [[0., 1.], [2., 3.]])])
def test_uint8_img_to_float_img(ui_tfi, dev_str, call):
    # smoke test
    if call is helpers.tf_graph_call:
        # tensorflow tensors cannot be cast to numpy arrays in graph mode
        pytest.skip()
    uint8_img, true_float_img = ui_tfi
    uint8_img = ivy.array(uint8_img, 'uint8', dev_str)
    true_float_img = np.array(true_float_img)
    float_img = ivy.uint8_img_to_float_img(uint8_img)
    # type test
    assert ivy.is_array(float_img)
    # cardinality test
    assert float_img.shape == true_float_img.shape
    # value test
    float_img_np = call(ivy.uint8_img_to_float_img, uint8_img)
    assert np.allclose(float_img_np, true_float_img)
    # compilation test
    if call in [helpers.torch_call]:
        # torch device cannot be assigned value of string while scripting
        return
    helpers.assert_compilable(ivy.uint8_img_to_float_img)


# random_crop
@pytest.mark.parametrize(
    "xshp_n_cs", [([2, 5, 6, 3], [2, 2])])
def test_random_crop(xshp_n_cs, dev_str, call):
    # seed
    ivy.seed(0)
    np.random.seed(0)
    # smoke test
    x_shape, crop_size = xshp_n_cs
    batch_size = x_shape[0]
    x_size = reduce(mul, x_shape[1:], 1)
    x = ivy.einops_repeat(ivy.reshape(ivy.arange(x_size), x_shape[1:]), '... -> b ...', b=batch_size)
    cropped = ivy.random_crop(x, crop_size)
    # type test
    assert ivy.is_array(cropped)
    # cardinality test
    true_shape = [item for item in x_shape]
    true_shape[-3] = crop_size[0]
    true_shape[-2] = crop_size[1]
    assert list(cropped.shape) == true_shape
    # value test
    assert np.allclose(ivy.to_numpy(x[0]), ivy.to_numpy(x[1]))
    cropped = call(ivy.random_crop, x, crop_size)
    assert not np.allclose(cropped[0], cropped[1])
    # compilation test
    if call in [helpers.torch_call]:
        # reduce(mul) used for flat batch size computation is not torch jit compilable
        return
    helpers.assert_compilable(ivy.random_crop)
