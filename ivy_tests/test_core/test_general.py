"""
Collection of tests for templated general functions
"""

# global
import math
import time
import einops
import pytest
import threading
import numpy as np
from numbers import Number
from collections.abc import Sequence
import torch.multiprocessing as multiprocessing

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# Helpers #
# --------#

def _get_shape_of_list(lst, shape=()):
    if not lst:
        return []
    if not isinstance(lst, Sequence):
        return shape
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)
    shape += (len(lst),)
    shape = _get_shape_of_list(lst[0], shape)
    return shape


# Tests #
# ------#

# set_framework
@pytest.mark.parametrize(
    "fw_str", ['numpy', 'jax', 'tensorflow', 'torch', 'mxnd'])
def test_set_framework(fw_str, dev_str, call):
    ivy.set_framework(fw_str)
    ivy.unset_framework()


# array
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "from_numpy", [True, False])
def test_array(object_in, dtype_str, from_numpy, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.array(object_in, dtype_str, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(call(ivy.array, object_in, dtype_str, dev_str), np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.array)


# array_equal
@pytest.mark.parametrize(
    "x0_n_x1_n_res", [([0.], [0.], True), ([0.], [1.], False),
                      ([[0.], [1.]], [[0.], [1.]], True),
                      ([[0.], [1.]], [[1.], [2.]], False)])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_array_equal(x0_n_x1_n_res, dtype_str, dev_str, call):
    if call in [helpers.mx_call] and dtype_str in ['int16', 'bool']:
        # mxnet does not support int16, and does not support bool for broadcast_equal method used
        pytest.skip()
    x0, x1, true_res = x0_n_x1_n_res
    # smoke test
    x0 = ivy.array(x0, dtype_str, dev_str)
    x1 = ivy.array(x1, dtype_str, dev_str)
    res = ivy.array_equal(x0, x1)
    # type test
    assert ivy.is_array(x0)
    assert ivy.is_array(x1)
    assert isinstance(res, bool) or ivy.is_array(res)
    # value test
    assert res == true_res
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.array_equal)


# equal
@pytest.mark.parametrize(
    "x0_n_x1_n_x2_em_n_res", [([0.], [0.], [0.], False, True),
                              ([0.], [1.], [0.], False, False),
                              ([0.], [1.], [0.], True, [[True, False, True],
                                                        [False, True, False],
                                                        [True, False, True]]),
                              ({'a': 0}, {'a': 0}, {'a': 1}, True, [[True, True, False],
                                                                    [True, True, False],
                                                                    [False, False, True]])])
@pytest.mark.parametrize(
    "to_array", [True, False])
def test_equal(x0_n_x1_n_x2_em_n_res, to_array, dev_str, call):
    x0, x1, x2, equality_matrix, true_res = x0_n_x1_n_x2_em_n_res
    # smoke test
    if isinstance(x0, list) and to_array:
        x0 = ivy.array(x0, dev_str=dev_str)
        x1 = ivy.array(x1, dev_str=dev_str)
        x2 = ivy.array(x2, dev_str=dev_str)
    res = ivy.equal(x0, x1, x2, equality_matrix=equality_matrix)
    # value test
    if equality_matrix:
        assert np.array_equal(ivy.to_numpy(res), np.array(true_res))
    else:
        assert res == true_res
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support variable number of input arguments
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.equal)


# to_numpy
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_numpy(object_in, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_numpy() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_numpy(tensor_fn(object_in, dtype_str, dev_str))
    # type test
    assert isinstance(ret, np.ndarray)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(ivy.to_numpy(tensor_fn(object_in, dtype_str, dev_str)), np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.to_numpy)


# to_scalar
@pytest.mark.parametrize(
    "object_in", [[0.], [[[1]]], [True], [[1.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_scalar(object_in, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_scalar() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_scalar(tensor_fn(object_in, dtype_str, dev_str))
    true_val = ivy.to_numpy(ivy.array(object_in, dtype_str=dtype_str)).item()
    # type test
    assert isinstance(ret, type(true_val))
    # value test
    assert ivy.to_scalar(tensor_fn(object_in, dtype_str, dev_str)) == true_val
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support scalar conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.to_scalar)


# to_list
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_list(object_in, dtype_str, tensor_fn, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_list(tensor_fn(object_in, dtype_str, dev_str))
    # type test
    assert isinstance(ret, list)
    # cardinality test
    assert _get_shape_of_list(ret) == _get_shape_of_list(object_in)
    # value test
    assert np.allclose(np.asarray(ivy.to_list(tensor_fn(object_in, dtype_str, dev_str))),
                       np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support list conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.to_list)


# shape
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_shape(object_in, dtype_str, as_tensor, tensor_fn, dev_str, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.shape(tensor_fn(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert ivy.is_array(ret)
    else:
        assert isinstance(ret, tuple)
        ret = ivy.array(ret)
    # cardinality test
    assert ret.shape[0] == len(np.asarray(object_in).shape)
    # value test
    assert np.array_equal(ivy.to_numpy(ret), np.asarray(np.asarray(object_in).shape, np.int32))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Union
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.shape)


# get_num_dims
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_get_num_dims(object_in, dtype_str, as_tensor, tensor_fn, dev_str, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.get_num_dims(tensor_fn(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert ivy.is_array(ret)
    else:
        assert isinstance(ret, int)
        ret = ivy.array(ret)
    # cardinality test
    assert list(ret.shape) == []
    # value test
    assert np.array_equal(ivy.to_numpy(ret), np.asarray(len(np.asarray(object_in).shape), np.int32))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Union
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.shape)


# minimum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_minimum(xy, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype_str, dev_str)
    y = tensor_fn(xy[1], dtype_str, dev_str)
    ret = ivy.minimum(x, y)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.minimum, x, y), np.asarray(ivy.numpy.minimum(ivy.to_numpy(x), ivy.to_numpy(y))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.minimum)


# maximum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_maximum(xy, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype_str, dev_str)
    y = tensor_fn(xy[1], dtype_str, dev_str)
    ret = ivy.maximum(x, y)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.maximum, x, y), np.asarray(ivy.numpy.maximum(ivy.to_numpy(x), ivy.to_numpy(y))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.maximum)


# clip
@pytest.mark.parametrize(
    "x_min_n_max", [(-0.5, 0., 1.5), ([1.7], [0.5], [1.1]), ([[0.8, 2.2], [1.5, 0.2]], 0.2, 1.4),
                    ([[0.8, 2.2], [1.5, 0.2]], [[1., 1.], [1., 1.]], [[1.1, 2.], [1.1, 2.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_clip(x_min_n_max, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_min_n_max[0], Number) or isinstance(x_min_n_max[1], Number) or isinstance(x_min_n_max[2], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_min_n_max[0], dtype_str, dev_str)
    min_val = tensor_fn(x_min_n_max[1], dtype_str, dev_str)
    max_val = tensor_fn(x_min_n_max[2], dtype_str, dev_str)
    if ((min_val.shape != [] and min_val.shape != [1]) or (max_val.shape != [] and max_val.shape != [1]))\
            and call in [helpers.mx_call]:
        # mxnet only supports numbers or 0 or 1 dimensional arrays for min and max while performing clip
        pytest.skip()
    ret = ivy.clip(x, min_val, max_val)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    max_shape = max([x.shape, min_val.shape, max_val.shape], key=lambda x_: len(x_))
    assert ret.shape == max_shape
    # value test
    assert np.array_equal(call(ivy.clip, x, min_val, max_val),
                          np.asarray(ivy.numpy.clip(ivy.to_numpy(x), ivy.to_numpy(min_val), ivy.to_numpy(max_val))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.clip)


# clip_vector_norm
@pytest.mark.parametrize(
    "x_max_norm_n_p_val_clipped",
    [(-0.5, 0.4, 2., -0.4), ([1.7], 1.5, 3., [1.5]),
     ([[0.8, 2.2], [1.5, 0.2]], 4., 1., [[0.6808511, 1.8723406], [1.2765958, 0.17021278]]),
     ([[0.8, 2.2], [1.5, 0.2]], 2.5, 2., [[0.71749604, 1.9731141], [1.345305, 0.17937401]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_clip_vector_norm(x_max_norm_n_p_val_clipped, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_max_norm_n_p_val_clipped[0], dtype_str, dev_str)
    max_norm = x_max_norm_n_p_val_clipped[1]
    p_val = x_max_norm_n_p_val_clipped[2]
    clipped = x_max_norm_n_p_val_clipped[3]
    ret = ivy.clip_vector_norm(x, max_norm, p_val)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (x.shape if len(x.shape) else (1,))
    # value test
    assert np.allclose(call(ivy.clip_vector_norm, x, max_norm, p_val), np.array(clipped))
    # compilation test
    if call is helpers.torch_call:
        # pytorch jit cannot compile global variables, in this case MIN_DENOMINATOR
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.clip_vector_norm)


# round
@pytest.mark.parametrize(
    "x_n_x_rounded", [(-0.51, -1), ([1.7], [2.]), ([[0.8, 2.2], [1.51, 0.2]], [[1., 2.], [2., 0.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_round(x_n_x_rounded, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_rounded[0], Number) or isinstance(x_n_x_rounded[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_rounded[0], dtype_str, dev_str)
    ret = ivy.round(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.array_equal(call(ivy.round, x), np.array(x_n_x_rounded[1]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.round)


# floormod
@pytest.mark.parametrize(
    "x_n_divisor_n_x_floormod", [(2.5, 2., 0.5), ([10.7], [5.], [0.7]),
                                 ([[0.8, 2.2], [1.7, 0.2]], [[0.3, 0.5], [0.4, 0.11]], [[0.2, 0.2], [0.1, 0.09]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_floormod(x_n_divisor_n_x_floormod, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_divisor_n_x_floormod[0], Number) or isinstance(x_n_divisor_n_x_floormod[1], Number) or
            isinstance(x_n_divisor_n_x_floormod[2], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_divisor_n_x_floormod[0], dtype_str, dev_str)
    divisor = ivy.array(x_n_divisor_n_x_floormod[1], dtype_str, dev_str)
    ret = ivy.floormod(x, divisor)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floormod, x, divisor), np.array(x_n_divisor_n_x_floormod[2]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.floormod)


# floor
@pytest.mark.parametrize(
    "x_n_x_floored", [(2.5, 2.), ([10.7], [10.]), ([[3.8, 2.2], [1.7, 0.2]], [[3., 2.], [1., 0.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_floor(x_n_x_floored, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_floored[0], Number) or isinstance(x_n_x_floored[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_floored[0], dtype_str, dev_str)
    ret = ivy.floor(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floor, x), np.array(x_n_x_floored[1]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.floor)


# ceil
@pytest.mark.parametrize(
    "x_n_x_ceiled", [(2.5, 3.), ([10.7], [11.]), ([[3.8, 2.2], [1.7, 0.2]], [[4., 3.], [2., 1.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_ceil(x_n_x_ceiled, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_ceiled[0], Number) or isinstance(x_n_x_ceiled[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_ceiled[0], dtype_str, dev_str)
    ret = ivy.ceil(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.ceil, x), np.array(x_n_x_ceiled[1]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.ceil)


# abs
@pytest.mark.parametrize(
    "x_n_x_absed", [(-2.5, 2.5), ([-10.7], [10.7]), ([[-3.8, 2.2], [1.7, -0.2]], [[3.8, 2.2], [1.7, 0.2]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_abs(x_n_x_absed, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_absed[0], Number) or isinstance(x_n_x_absed[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_absed[0], dtype_str, dev_str)
    ret = ivy.abs(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.abs, x), np.array(x_n_x_absed[1]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.abs)


# argmax
@pytest.mark.parametrize(
    "x_n_axis_x_argmax", [([-0.3, 0.1], None, [1]), ([[1.3, 2.6], [2.3, 2.5]], 0, [1, 0]),
                          ([[1.3, 2.6], [2.3, 2.5]], 1, [1, 1])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_argmax(x_n_axis_x_argmax, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = ivy.array(x_n_axis_x_argmax[0], dtype_str, dev_str)
    axis = x_n_axis_x_argmax[1]
    ret = ivy.argmax(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert tuple(ret.shape) == (len(x.shape),)
    # value test
    assert np.allclose(call(ivy.argmax, x, axis), np.array(x_n_axis_x_argmax[2]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.argmax)


# argmin
@pytest.mark.parametrize(
    "x_n_axis_x_argmin", [([-0.3, 0.1], None, [0]), ([[1.3, 2.6], [2.3, 2.5]], 0, [0, 1]),
                          ([[1.3, 2.6], [2.3, 2.5]], 1, [0, 0])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_argmin(x_n_axis_x_argmin, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x_n_axis_x_argmin[0], dtype_str, dev_str)
    axis = x_n_axis_x_argmin[1]
    ret = ivy.argmin(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert tuple(ret.shape) == (len(x.shape),)
    # value test
    assert np.allclose(call(ivy.argmin, x, axis), np.array(x_n_axis_x_argmin[2]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.argmin)


# argsort
@pytest.mark.parametrize(
    "x_n_axis_x_argsort", [([1, 10, 26.9, 2.8, 166.32, 62.3], -1, [0, 3, 1, 2, 5, 4])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_argsort(x_n_axis_x_argsort, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x_n_axis_x_argsort[0], dtype_str, dev_str)
    axis = x_n_axis_x_argsort[1]
    ret = ivy.argsort(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert tuple(ret.shape) == (6,)
    # value test
    assert np.allclose(call(ivy.argsort, x, axis), np.array(x_n_axis_x_argsort[2]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.argsort)


# cast
@pytest.mark.parametrize(
    "object_in", [[1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "starting_dtype_str", ['float32', 'int32', 'bool'])
@pytest.mark.parametrize(
    "target_dtype_str", ['float32', 'int32', 'bool'])
def test_cast(object_in, starting_dtype_str, target_dtype_str, dev_str, call):
    # smoke test
    x = ivy.array(object_in, starting_dtype_str, dev_str)
    ret = ivy.cast(x, target_dtype_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert ivy.dtype_str(ret) == target_dtype_str
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cast)


# arange
@pytest.mark.parametrize(
    "stop_n_start_n_step", [[10, None, None], [10, 2, None], [10, 2, 2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_arange(stop_n_start_n_step, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    stop, start, step = stop_n_start_n_step
    if (isinstance(stop, Number) or isinstance(start, Number) or isinstance(step, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    if tensor_fn == helpers.var_fn and call is helpers.torch_call:
        # pytorch does not support arange using variables as input
        pytest.skip()
    args = list()
    if stop:
        stop = tensor_fn(stop, dtype_str, dev_str)
        args.append(stop)
    if start:
        start = tensor_fn(start, dtype_str, dev_str)
        args.append(start)
    if step:
        step = tensor_fn(step, dtype_str, dev_str)
        args.append(step)
    ret = ivy.arange(*args, dtype_str=dtype_str, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (int((ivy.to_list(stop) -
                              (ivy.to_list(start) if start else 0))/(ivy.to_list(step) if step else 1)),)
    # value test
    assert np.array_equal(call(ivy.arange, *args, dtype_str=dtype_str, dev_str=dev_str),
                          np.asarray(ivy.numpy.arange(*[ivy.to_numpy(arg) for arg in args], dtype_str=dtype_str)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Number type, or Union for Union[float, int] etc.
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.arange)


# linspace
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_axis", [[1, 10, 100, None], [[[0., 1., 2.]], [[1., 2., 3.]], 150, -1],
                                  [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, -2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linspace(start_n_stop_n_num_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    start, stop, num, axis = start_n_stop_n_num_n_axis
    if (isinstance(start, Number) or isinstance(stop, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype_str, dev_str)
    stop = tensor_fn(stop, dtype_str, dev_str)
    ret = ivy.linspace(start, stop, num, axis, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(call(ivy.linspace, start, stop, num, axis, dev_str=dev_str),
                       np.asarray(ivy.numpy.linspace(ivy.to_numpy(start), ivy.to_numpy(stop), num, axis)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.linspace)


# logspace
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_base_n_axis", [[1, 10, 100, 10., None], [[[0., 1., 2.]], [[1., 2., 3.]], 150, 2., -1],
                                         [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, 5., -2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logspace(start_n_stop_n_num_n_base_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    start, stop, num, base, axis = start_n_stop_n_num_n_base_n_axis
    if (isinstance(start, Number) or isinstance(stop, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype_str, dev_str)
    stop = tensor_fn(stop, dtype_str, dev_str)
    ret = ivy.logspace(start, stop, num, base, axis, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(call(ivy.logspace, start, stop, num, base, axis, dev_str=dev_str),
                       ivy.numpy.logspace(ivy.to_numpy(start), ivy.to_numpy(stop), num, base, axis))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.logspace)


# concatenate
@pytest.mark.parametrize(
    "x1_n_x2_n_axis", [(1, 10, 0), ([[0., 1., 2.]], [[1., 2., 3.]], 0), ([[0., 1., 2.]], [[1., 2., 3.]], 1),
                       ([[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], -1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_concatenate(x1_n_x2_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2, axis = x1_n_x2_n_axis
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.concatenate((x1, x2), axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    axis_val = (axis % len(x1.shape) if (axis is not None and len(x1.shape) != 0) else len(x1.shape) - 1)
    if x1.shape == ():
        expected_shape = (2,)
    else:
        expected_shape = tuple([item * 2 if i == axis_val else item for i, item in enumerate(x1.shape)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.concatenate, [x1, x2], axis),
                       np.asarray(ivy.numpy.concatenate([ivy.to_numpy(x1), ivy.to_numpy(x2)], axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.concatenate)


# flip
@pytest.mark.parametrize(
    "x_n_axis_n_bs", [(1, 0, None), ([[0., 1., 2.]], None, (1, 3)), ([[0., 1., 2.]], 1, (1, 3)),
                       ([[[-0.1471, 0.4477, 0.2214]]], None, None)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_flip(x_n_axis_n_bs, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis, bs = x_n_axis_n_bs
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.flip(x, axis, bs)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.flip, x, axis, bs), np.asarray(ivy.numpy.flip(ivy.to_numpy(x), axis, bs)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.flip)


# stack
@pytest.mark.parametrize(
    "xs_n_axis", [((1, 0), -1), (([[0., 1., 2.]], [[3., 4., 5.]]), 0), (([[0., 1., 2.]], [[3., 4., 5.]]), 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_stack(xs_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    (x1, x2), axis = xs_n_axis
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.stack((x1, x2), axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    axis_val = (axis % len(x1.shape) if (axis is not None and len(x1.shape) != 0) else len(x1.shape) - 1)
    if x1.shape == ():
        expected_shape = (2,)
    else:
        expected_shape = list(x1.shape)
        expected_shape.insert(axis_val, 2)
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.stack, (x1, x2), axis),
                       np.asarray(ivy.numpy.stack((ivy.to_numpy(x1), ivy.to_numpy(x2)), axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.stack)


# unstack
@pytest.mark.parametrize(
    "x_n_axis", [(1, -1), ([[0., 1., 2.]], 0), ([[0., 1., 2.]], 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_unstack(x_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.unstack(x, axis)
    # type test
    assert isinstance(ret, list)
    # cardinality test
    axis_val = (axis % len(x.shape) if (axis is not None and len(x.shape) != 0) else len(x.shape) - 1)
    if x.shape == ():
        expected_shape = ()
    else:
        expected_shape = list(x.shape)
        expected_shape.pop(axis_val)
    assert ret[0].shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.unstack, x, axis), np.asarray(ivy.numpy.unstack(ivy.to_numpy(x), axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.unstack)


# split
@pytest.mark.parametrize(
    "x_n_noss_n_axis_n_wr", [(1, 1, -1, False),
                             ([[0., 1., 2., 3.]], 2, 1, False),
                             ([[0., 1., 2.], [3., 4., 5.]], 2, 0, False),
                             ([[0., 1., 2.], [3., 4., 5.]], 2, 1, True),
                             ([[0., 1., 2.], [3., 4., 5.]], [2, 1], 1, False)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split(x_n_noss_n_axis_n_wr, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, num_or_size_splits, axis, with_remainder = x_n_noss_n_axis_n_wr
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    if (isinstance(num_or_size_splits, list) or with_remainder) and call is helpers.mx_call:
        # mxnet does not support split method with section sizes, only num_sections is supported.
        # This only explains why remainders aren't supported, as this uses the same underlying mechanism.
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.split(x, num_or_size_splits, axis, with_remainder)
    # type test
    assert isinstance(ret, list)
    # cardinality test
    axis_val = (axis % len(x.shape) if (axis is not None and len(x.shape) != 0) else len(x.shape) - 1)
    if x.shape == ():
        expected_shape = ()
    elif isinstance(num_or_size_splits, int):
        expected_shape = tuple([math.ceil(item/num_or_size_splits) if i == axis_val else item
                                for i, item in enumerate(x.shape)])
    else:
        expected_shape = tuple([num_or_size_splits[0] if i == axis_val else item for i, item in enumerate(x.shape)])
    assert ret[0].shape == expected_shape
    # value test
    pred_split = call(ivy.split, x, num_or_size_splits, axis, with_remainder)
    true_split = ivy.numpy.split(ivy.to_numpy(x), num_or_size_splits, axis, with_remainder)
    for pred, true in zip(pred_split, true_split):
        assert np.allclose(pred, true)
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.split)


# repeat
@pytest.mark.parametrize(
    "x_n_reps_n_axis", [(1, [1], 0), (1, 2, -1), (1, [2], None), ([[0., 1., 2., 3.]], (2, 1, 0, 3), -1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_repeat(x_n_reps_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, reps_raw, axis = x_n_reps_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    if not isinstance(reps_raw, int) and call is helpers.mx_call:
        # mxnet repeat only supports integer repeats
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    x_shape = list(x.shape)
    if call not in [helpers.jnp_call, helpers.torch_call]:
        # jax and pytorch repeat do not support repeats specified as lists
        ret_from_list = ivy.repeat(x, reps_raw, axis)
    reps = ivy.array(reps_raw, 'int32', dev_str)
    if call is helpers.mx_call:
        # mxnet only supports repeats defined as as int
        ret = ivy.repeat(x, reps_raw, axis)
    else:
        ret = ivy.repeat(x, reps, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if x.shape == ():
        expected_shape = [reps_raw] if isinstance(reps_raw, int) else list(reps_raw)
    else:
        axis_wrapped = axis % len(x_shape)
        expected_shape = x_shape[0:axis_wrapped] + [sum(reps_raw)] + x_shape[axis_wrapped+1:]
    assert list(ret.shape) == expected_shape
    # value test
    if call is helpers.mx_call:
        # mxnet only supports repeats defined as as int
        assert np.allclose(call(ivy.repeat, x, reps_raw, axis),
                           np.asarray(ivy.numpy.repeat(ivy.to_numpy(x), ivy.to_numpy(reps), axis)))
    else:
        assert np.allclose(call(ivy.repeat, x, reps, axis),
                           np.asarray(ivy.numpy.repeat(ivy.to_numpy(x), ivy.to_numpy(reps), axis)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not union of types
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.repeat)


# tile
@pytest.mark.parametrize(
    "x_n_reps", [(1, [1]), (1, 2), (1, [2]), ([[0., 1., 2., 3.]], (2, 1))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_tile(x_n_reps, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, reps_raw = x_n_reps
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret_from_list = ivy.tile(x, reps_raw)
    reps = ivy.array(reps_raw, 'int32', dev_str)
    ret = ivy.tile(x, reps)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if x.shape == ():
        expected_shape = tuple(reps_raw) if isinstance(reps_raw, list) else (reps_raw,)
    else:
        expected_shape = tuple([int(item * rep) for item, rep in zip(x.shape, reps_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.tile, x, reps),
                       np.asarray(ivy.numpy.tile(ivy.to_numpy(x), ivy.to_numpy(reps))))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.tile)


# zero_pad
@pytest.mark.parametrize(
    "x_n_pw", [(1, [[1, 1]]), (1, [[0, 0]]), ([[0., 1., 2., 3.]], [[0, 1], [1, 2]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zero_pad(x_n_pw, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, pw_raw = x_n_pw
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret_from_list = ivy.zero_pad(x, pw_raw)
    pw = ivy.array(pw_raw, 'int32', dev_str)
    ret = ivy.zero_pad(x, pw)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else x.shape
    expected_shape = tuple([int(item + pw_[0] + pw_[1]) for item, pw_ in zip(x_shape, pw_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.zero_pad, x, pw), ivy.numpy.zero_pad(ivy.to_numpy(x), ivy.to_numpy(pw)))
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.zero_pad)


# fourier_encode
@pytest.mark.parametrize(
    "x_n_mf_n_nb_n_gt", [([2.], 4., 4, [[2.,  1.7484555e-07,  0.99805772, -0.52196848,
                                         3.4969111e-07, 1., -0.062295943, -0.85296476, 1.]]),
                         ([[0.5, 1.5, 2.5, 3.5]], 8., 6,
                              [[[5.0000000e-01, 1.0000000e+00, 8.7667871e-01, 3.9555991e-01,
                                 -4.5034310e-01, -9.9878132e-01, 1.7484555e-07, -4.3711388e-08,
                                 -4.8107630e-01, -9.1844016e-01, -8.9285558e-01, 4.9355008e-02, 1.0000000e+00],
                                [1.5000000e+00, -1.0000000e+00, -6.5104485e-02,  9.3911028e-01, -9.8569500e-01,
                                 9.8904943e-01,  4.7699523e-08,  1.1924881e-08, 9.9787843e-01, -3.4361595e-01,
                                 -1.6853882e-01, -1.4758460e-01, 1.0000000e+00],
                                [2.5000000e+00, 1.0000000e+00, -8.0673981e-01, 8.9489913e-01, -7.2141492e-01,
                                 -9.6968061e-01, 1.3510650e-06, -3.3776624e-07, -5.9090680e-01, 4.4626841e-01,
                                 6.9250309e-01,  2.4437571e-01, 1.0000000e+00],
                                [3.5000000e+00, -1.0000000e+00, 9.3175411e-01, 2.9059762e-01, 1.2810004e-01,
                                 9.4086391e-01,  2.6544303e-06,  6.6360758e-07, -3.6308998e-01, 9.5684534e-01,
                                 9.9176127e-01, -3.3878478e-01, 1.0000000e+00]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_fourier_encode(x_n_mf_n_nb_n_gt, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, max_freq, num_bands, ground_truth = x_n_mf_n_nb_n_gt
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.fourier_encode(x, max_freq, num_bands)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else list(x.shape)
    expected_shape = x_shape + [1 + 2*num_bands]
    assert list(ret.shape) == expected_shape
    # value test
    assert np.allclose(call(ivy.fourier_encode, x, max_freq, num_bands), np.array(ground_truth), atol=1e-5)
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.fourier_encode)


# constant_pad
@pytest.mark.parametrize(
    "x_n_pw_n_val", [(1, [[1, 1]], 1.5), (1, [[0, 0]], -2.7), ([[0., 1., 2., 3.]], [[0, 1], [1, 2]], 11.)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_constant_pad(x_n_pw_n_val, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, pw_raw, val = x_n_pw_n_val
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret_from_list = ivy.constant_pad(x, pw_raw, val)
    pw = ivy.array(pw_raw, 'int32', dev_str)
    ret = ivy.constant_pad(x, pw, val)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else x.shape
    expected_shape = tuple([int(item + pw_[0] + pw_[1]) for item, pw_ in zip(x_shape, pw_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.constant_pad, x, pw, val),
                       np.asarray(ivy.numpy.constant_pad(ivy.to_numpy(x), ivy.to_numpy(pw), val)))
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.constant_pad)


# swapaxes
@pytest.mark.parametrize(
    "x_n_ax0_n_ax1", [([[1.]], 0, 1), ([[0., 1., 2., 3.]], 1, 0), ([[[0., 1., 2.], [3., 4., 5.]]], -2, -1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_swapaxes(x_n_ax0_n_ax1, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, ax0, ax1 = x_n_ax0_n_ax1
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.swapaxes(x, ax0, ax1)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    expected_shape = list(x.shape)
    expected_shape[ax0], expected_shape[ax1] = expected_shape[ax1], expected_shape[ax0]
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.swapaxes, x, ax0, ax1),
                       np.asarray(ivy.numpy.swapaxes(ivy.to_numpy(x), ax0, ax1)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.swapaxes)


# transpose
@pytest.mark.parametrize(
    "x_n_axes", [([[1.]], [1, 0]), ([[0., 1., 2., 3.]], [1, 0]), ([[[0., 1., 2.], [3., 4., 5.]]], [0, 2, 1])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_transpose(x_n_axes, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axes = x_n_axes
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.transpose(x, axes)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = x.shape
    assert ret.shape == tuple([x.shape[idx] for idx in axes])
    # value test
    assert np.allclose(call(ivy.transpose, x, axes), np.asarray(ivy.numpy.transpose(ivy.to_numpy(x), axes)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.transpose)


# expand_dims
@pytest.mark.parametrize(
    "x_n_axis", [(1., 0), (1., -1), ([1.], 0), ([[0., 1., 2., 3.]], -2), ([[[0., 1., 2.], [3., 4., 5.]]], -3)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_expand_dims(x_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.expand_dims(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    expected_shape = list(x.shape)
    expected_shape.insert(axis, 1)
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.expand_dims, x, axis), np.asarray(ivy.numpy.expand_dims(ivy.to_numpy(x), axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.expand_dims)


# where
@pytest.mark.parametrize(
    "cond_n_x1_n_x2", [(True, 2., 3.), (0., 2., 3.), ([True], [2.], [3.]), ([[0.]], [[2., 3.]], [[4., 5.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_where(cond_n_x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    cond, x1, x2 = cond_n_x1_n_x2
    if (isinstance(cond, Number) or isinstance(x1, Number) or isinstance(x2, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    cond = tensor_fn(cond, dtype_str, dev_str)
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.where(cond, x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.where, cond, x1, x2),
                       np.asarray(ivy.numpy.where(ivy.to_numpy(cond), ivy.to_numpy(x1), ivy.to_numpy(x2))))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.where)


# indices_where
@pytest.mark.parametrize(
    "x", [[True], [[0., 1.], [2., 3.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_indices_where(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.indices_where(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert len(ret.shape) == 2
    assert ret.shape[-1] == len(x.shape)
    # value test
    assert np.allclose(call(ivy.indices_where, x), np.asarray(ivy.numpy.indices_where(ivy.to_numpy(x))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.indices_where)


# isnan
@pytest.mark.parametrize(
    "x_n_res", [([True], [False]),
                ([[0., float('nan')], [float('nan'), 3.]],
                 [[False, True], [True, False]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_isnan(x_n_res, dtype_str, tensor_fn, dev_str, call):
    x, res = x_n_res
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.isnan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.isnan, x), res)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.isnan)


# reshape
@pytest.mark.parametrize(
    "x_n_shp", [(1., (1, 1)), (1., 1), (1., []), ([[1.]], []), ([[0., 1.], [2., 3.]], (1, 4, 1))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reshape(x_n_shp, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, new_shape = x_n_shp
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.reshape(x, new_shape)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == ((new_shape,) if isinstance(new_shape, int) else tuple(new_shape))
    # value test
    assert np.allclose(call(ivy.reshape, x, new_shape), np.asarray(ivy.numpy.reshape(ivy.to_numpy(x), new_shape)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.reshape)


# broadcast_to
@pytest.mark.parametrize(
    "x_n_shp", [([1.], (2, 1)), ([[0., 1.], [2., 3.]], (10, 2, 2))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_broadcast_to(x_n_shp, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, new_shape = x_n_shp
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.broadcast_to(x, new_shape)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert len(ret.shape) == len(new_shape)
    # value test
    assert np.allclose(call(ivy.broadcast_to, x, new_shape),
                       np.asarray(ivy.numpy.broadcast_to(ivy.to_numpy(x), new_shape)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.broadcast_to)


# squeeze
@pytest.mark.parametrize(
    "x_n_axis", [(1., 0), (1., -1), ([[1.]], None), ([[[0.], [1.]], [[2.], [3.]]], -1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_squeeze(x_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.squeeze(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if axis is None:
        expected_shape = [item for item in x.shape if item != 1]
    elif x.shape == ():
        expected_shape = []
    else:
        expected_shape = list(x.shape)
        expected_shape.pop(axis)
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.squeeze, x, axis), np.asarray(ivy.numpy.squeeze(ivy.to_numpy(x), axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.squeeze)


# zeros
@pytest.mark.parametrize(
    "shape", [(), (1, 2, 3), tuple([1]*10)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zeros(shape, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ret = ivy.zeros(shape, dtype_str, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(shape)
    # value test
    assert np.allclose(call(ivy.zeros, shape, dtype_str, dev_str), np.asarray(ivy.numpy.zeros(shape, dtype_str)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.zeros)


# zeros_like
@pytest.mark.parametrize(
    "x", [1, [1], [[1], [2], [3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zeros_like(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.zeros_like(x, dtype_str, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.zeros_like, x, dtype_str, dev_str),
                       np.asarray(ivy.numpy.zeros_like(ivy.to_numpy(x), dtype_str)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.zeros_like)


# ones
@pytest.mark.parametrize(
    "shape", [(), (1, 2, 3), tuple([1]*10)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_ones(shape, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ret = ivy.ones(shape, dtype_str, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(shape)
    # value test
    assert np.allclose(call(ivy.ones, shape, dtype_str, dev_str), np.asarray(ivy.numpy.ones(shape, dtype_str)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.ones)


# ones_like
@pytest.mark.parametrize(
    "x", [1, [1], [[1], [2], [3]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_ones_like(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.ones_like(x, dtype_str, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.ones_like, x, dtype_str, dev_str),
                       np.asarray(ivy.numpy.ones_like(ivy.to_numpy(x), dtype_str)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.ones_like)


# one_hot
@pytest.mark.parametrize(
    "ind_n_depth", [([0], 1), ([0, 1, 2], 3), ([[1, 3], [0, 0], [8, 4], [7, 9]], 10)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_one_hot(ind_n_depth, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ind, depth = ind_n_depth
    if isinstance(ind, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ind = ivy.array(ind, 'int32', dev_str)
    ret = ivy.one_hot(ind, depth, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == ind.shape + (depth,)
    # value test
    assert np.allclose(call(ivy.one_hot, ind, depth, dev_str),
                       np.asarray(ivy.numpy.one_hot(ivy.to_numpy(ind), depth)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.one_hot)


# cross
@pytest.mark.parametrize(
    "x1_n_x2", [([0., 1., 2.], [3., 4., 5.]), ([[0., 1., 2.], [2., 1., 0.]], [[3., 4., 5.], [5., 4., 3.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cross(x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2 = x1_n_x2
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = ivy.array(x1, dtype_str, dev_str)
    x2 = ivy.array(x2, dtype_str, dev_str)
    ret = ivy.cross(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.cross, x1, x2), np.asarray(ivy.numpy.cross(ivy.to_numpy(x1), ivy.to_numpy(x2))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cross)


# matmul
@pytest.mark.parametrize(
    "x1_n_x2", [([[0., 1., 2.]], [[3.], [4.], [5.]]), ([[0., 1., 2.], [2., 1., 0.]], [[3., 4.], [5., 5.], [4., 3.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_matmul(x1_n_x2, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2 = x1_n_x2
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = ivy.array(x1, dtype_str, dev_str)
    x2 = ivy.array(x2, dtype_str, dev_str)
    ret = ivy.matmul(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape[:-1] + (x2.shape[-1],)
    # value test
    assert np.allclose(call(ivy.matmul, x1, x2), np.asarray(ivy.numpy.matmul(ivy.to_numpy(x1), ivy.to_numpy(x2))))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.matmul)


# cumsum
@pytest.mark.parametrize(
    "x_n_axis", [([[0., 1., 2.]], -1), ([[0., 1., 2.], [2., 1., 0.]], 0), ([[0., 1., 2.], [2., 1., 0.]], 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cumsum(x_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype_str, dev_str)
    ret = ivy.cumsum(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cumsum, x, axis), np.asarray(ivy.numpy.cumsum(ivy.to_numpy(x), axis)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cumsum)


# cumprod
@pytest.mark.parametrize(
    "x_n_axis", [([[0., 1., 2.]], -1), ([[0., 1., 2.], [2., 1., 0.]], 0), ([[0., 1., 2.], [2., 1., 0.]], 1)])
@pytest.mark.parametrize(
    "exclusive", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cumprod(x_n_axis, exclusive, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype_str, dev_str)
    ret = ivy.cumprod(x, axis, exclusive)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cumprod, x, axis, exclusive),
                       np.asarray(ivy.numpy.cumprod(ivy.to_numpy(x), axis, exclusive)))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cumprod)


# identity
@pytest.mark.parametrize(
    "dim_n_bs", [(3, None), (1, (2, 3)), (5, (1, 2, 3))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_identity(dim_n_bs, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    dim, bs = dim_n_bs
    ret = ivy.identity(dim, dtype_str, bs, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (tuple(bs) if bs else ()) + (dim, dim)
    # value test
    assert np.allclose(call(ivy.identity, dim, dtype_str, bs, dev_str),
                       np.asarray(ivy.numpy.identity(dim, dtype_str, bs)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.identity)


# meshgrid
@pytest.mark.parametrize(
    "xs", [([1, 2, 3], [4, 5, 6]), ([1, 2, 3], [4, 5, 6, 7], [8, 9])])
@pytest.mark.parametrize(
    "indexing", ['xy', 'ij'])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_meshgrid(xs, indexing, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    xs_as_arrays = [ivy.array(x, 'int32', dev_str) for x in xs]
    rets = ivy.meshgrid(*xs_as_arrays, indexing=indexing)
    # type test
    for ret in rets:
        assert ivy.is_array(ret)
    # cardinality test
    target_shape = tuple([len(x) for x in xs])
    if indexing == 'xy':
        target_shape = (target_shape[1], target_shape[0]) + target_shape[2:]
    for ret in rets:
        assert ret.shape == target_shape
    # value test
    assert np.allclose(
        call(ivy.meshgrid, *xs_as_arrays, indexing=indexing),
        [np.asarray(i) for i in ivy.numpy.meshgrid(*[ivy.to_numpy(x) for x in xs_as_arrays], indexing=indexing)])
    # compilation test
    if call is helpers.torch_call:
        # torch scripting can't take variable number of arguments or use keyword-only arguments with defaults
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.meshgrid)


# scatter_flat
@pytest.mark.parametrize(
    "inds_n_upd_n_size", [([0, 4, 1, 2], [1, 2, 3, 4], 8), ([0, 4, 1, 2, 0], [1, 2, 3, 4, 5], 8)])
@pytest.mark.parametrize(
    "red", ['sum', 'min', 'max', 'replace'])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_flat(inds_n_upd_n_size, red, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (red == 'sum' or red == 'min' or red == 'max') and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    if red == 'replace' and call is not helpers.mx_call:
        # mxnet is the only backend which supports the replace reduction
        pytest.skip()
    inds, upd, size = inds_n_upd_n_size
    inds = ivy.array(inds, 'int32', dev_str)
    upd = tensor_fn(upd, dtype_str, dev_str)
    ret = ivy.scatter_flat(inds, upd, size, red, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (size,)
    if red == 'replace':
        return
    # value test
    assert np.allclose(call(ivy.scatter_flat, inds, upd, size, red, dev_str),
                       np.asarray(ivy.numpy.scatter_flat(ivy.to_numpy(inds), ivy.to_numpy(upd), size, red)))
    # compilation test
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.scatter_flat)


# scatter_nd
@pytest.mark.parametrize(
    "inds_n_upd_n_shape", [([[4], [3], [1], [7]], [9, 10, 11, 12], [8]), ([[0, 1, 2]], [1], [3, 3, 3]),
                           ([[0], [2]], [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                                         [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], [4, 4, 4])])
@pytest.mark.parametrize(
    "red", ['sum', 'min', 'max', 'replace'])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_nd(inds_n_upd_n_shape, red, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (red == 'sum' or red == 'min' or red == 'max') and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    if red == 'replace' and call is not helpers.mx_call:
        # mxnet is the only backend which supports the replace reduction
        pytest.skip()
    inds, upd, shape = inds_n_upd_n_shape
    inds = ivy.array(inds, 'int32', dev_str)
    upd = tensor_fn(upd, dtype_str, dev_str)
    ret = ivy.scatter_nd(inds, upd, shape, red, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(shape)
    if red == 'replace':
        return
    # value test
    assert np.allclose(call(ivy.scatter_nd, inds, upd, shape, red, dev_str),
                       np.asarray(ivy.numpy.scatter_nd(ivy.to_numpy(inds), ivy.to_numpy(upd), shape, red)))
    # compilation test
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.scatter_nd)


# gather
@pytest.mark.parametrize(
    "prms_n_inds_n_axis", [([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 4, 7], 0),
                           ([[1, 2], [3, 4]], [[0, 0], [1, 0]], 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gather(prms_n_inds_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    prms, inds, axis = prms_n_inds_n_axis
    prms = tensor_fn(prms, dtype_str, dev_str)
    inds = ivy.array(inds, 'int32', dev_str)
    ret = ivy.gather(prms, inds, axis, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == inds.shape
    # value test
    assert np.allclose(call(ivy.gather, prms, inds, axis, dev_str),
                       np.asarray(ivy.numpy.gather(ivy.to_numpy(prms), ivy.to_numpy(inds), axis)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.gather)


# gather_nd
@pytest.mark.parametrize(
    "prms_n_inds", [([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[0, 1], [1, 0]]),
                    ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[[0, 1]], [[1, 0]]]),
                    ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[[0, 1, 0]], [[1, 0, 1]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gather_nd(prms_n_inds, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    prms, inds = prms_n_inds
    prms = tensor_fn(prms, dtype_str, dev_str)
    inds = ivy.array(inds, 'int32', dev_str)
    ret = ivy.gather_nd(prms, inds, dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == inds.shape[:-1] + prms.shape[inds.shape[-1]:]
    # value test
    assert np.allclose(call(ivy.gather_nd, prms, inds, dev_str),
                       np.asarray(ivy.numpy.gather_nd(ivy.to_numpy(prms), ivy.to_numpy(inds))))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.gather_nd)


# linear_resample
@pytest.mark.parametrize(
    "x_n_samples_n_axis_n_y_true", [([[10., 9., 8.]], 9, -1, [[10., 9.75, 9.5, 9.25, 9., 8.75, 8.5, 8.25, 8.]]),
                                    ([[[10., 9.], [8., 7.]]], 5, -2,
                                     [[[10., 9.], [9.5, 8.5], [9., 8.], [8.5, 7.5], [8., 7.]]]),
                                    ([[[10., 9.], [8., 7.]]], 5, -1,
                                     [[[10., 9.75, 9.5, 9.25, 9.], [8., 7.75, 7.5, 7.25, 7.]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linear_resample(x_n_samples_n_axis_n_y_true, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, samples, axis, y_true = x_n_samples_n_axis_n_y_true
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.linear_resample(x, samples, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    x_pre_shape = x_shape[0:axis]
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis+1:]
    assert list(ret.shape) == x_pre_shape + [samples] + x_post_shape
    # value test
    y_true = np.array(y_true)
    y = call(ivy.linear_resample, x, samples, axis)
    assert np.allclose(y, y_true)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.linear_resample)


# exists
@pytest.mark.parametrize(
    "x", [[1.], None, [[10., 9., 8.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_exists(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str) if x is not None else None
    ret = ivy.exists(x)
    # type test
    assert isinstance(ret, bool)
    # value test
    y_true = x is not None
    assert ret == y_true
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.exists)


# default
@pytest.mark.parametrize(
    "x_n_dv", [([1.], [2.]), (None, [2.]), ([[10., 9., 8.]], [2.])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_default(x_n_dv, dtype_str, tensor_fn, dev_str, call):
    x, dv = x_n_dv
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str) if x is not None else None
    dv = tensor_fn(dv, dtype_str, dev_str)
    ret = ivy.default(x, dv)
    # type test
    assert ivy.is_array(ret)
    # value test
    y_true = ivy.to_numpy(x if x is not None else dv)
    assert np.allclose(call(ivy.default, x, dv), y_true)
    # compilation test
    if call is helpers.torch_call:
        # try-except blocks are not jit compilable in pytorch
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.default)


# dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dev(x)
    # type test
    assert isinstance(ret, ivy.Device)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev)


# to_dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_to_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    x_on_dev = ivy.to_dev(x, dev_str)
    dev_from_new_x = ivy.dev(x)
    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert '/' + ':'.join(dev_from_new_x[1:].split(':')[-2:]) == '/' + ':'.join(dev[1:].split(':')[-2:])
    elif call is helpers.torch_call:
        assert dev_from_new_x.type == dev.type
    else:
        assert dev_from_new_x == dev
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not handle converting string to device
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.to_dev)


# dev_to_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev_to_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    ret = ivy.dev_to_str(dev)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dev_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev_to_str)


# str_to_dev
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_str_to_dev(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dev = ivy.dev(x)
    ret = ivy.str_to_dev(dev_str)
    # value test
    if call in [helpers.tf_call, helpers.tf_graph_call]:
        assert '/' + ':'.join(ret[1:].split(':')[-2:]) == '/' + ':'.join(dev[1:].split(':')[-2:])
    elif call is helpers.torch_call:
        assert ret.type == dev.type
    else:
        assert ret == dev
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not handle converting string to device
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.str_to_dev)


# dev_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_dev_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dev_str(x)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dev_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dev_str)


# memory_on_dev
@pytest.mark.parametrize(
    "dev_str_to_check", ['cpu', 'cpu:0', 'gpu:0'])
def test_memory_on_dev(dev_str_to_check, dev_str, call):
    if 'gpu' in dev_str_to_check and ivy.num_gpus() == 0:
        # cannot get amount of memory for gpu which is not present
        pytest.skip()
    ret = ivy.memory_on_dev(dev_str_to_check)
    # type test
    assert isinstance(ret, float)
    # value test
    assert 0 < ret < 64
    # compilation test
    if call is helpers.torch_call:
        # global variables aren't supported for pytorch scripting
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.memory_on_dev)


# dtype
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dtype(x)
    # type test
    assert isinstance(ret, ivy.Dtype)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dtype)


# dtype_to_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype_to_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.mx_call and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype_str in ['int64', 'float64']:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    dtype = ivy.dtype(x)
    ret = ivy.dtype_to_str(dtype)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dtype_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dtype_to_str)


# dtype_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype_str(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call is helpers.mx_call and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype_str in ['int64', 'float64']:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.dtype_str(x)
    # type test
    assert isinstance(ret, str)
    # value test
    assert ret == dtype_str
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.dtype_str)


# compile_fn
def _fn_1(x):
    return x**2


def _fn_2(x):
    return (x + 10)**0.5 - 5


@pytest.mark.parametrize(
    "x", [[1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_compile_fn(x, dtype_str, tensor_fn, dev_str, call):
    if ivy.wrapped_mode():
        # Wrapped mode does not yet support function compilation
        pytest.skip()
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    # function 1
    x = tensor_fn(x, dtype_str, dev_str)
    comp_fn = ivy.compile_fn(_fn_1)
    # type test
    assert callable(comp_fn)
    # value test
    non_compiled_return = _fn_1(x)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))

    # function 2
    x = tensor_fn(x, dtype_str, dev_str)
    comp_fn = ivy.compile_fn(_fn_2)
    # type test
    assert callable(comp_fn)
    # value test
    non_compiled_return = _fn_2(x)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))


@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 3])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = tensor_fn(x0, 'float32', dev_str)
    in1 = tensor_fn(x1, 'float32', dev_str)

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(func, [in0, in1], chunk_size, axis)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a), ivy.to_numpy(a_true))
    assert np.allclose(ivy.to_numpy(b), ivy.to_numpy(b_true))
    assert np.allclose(ivy.to_numpy(c), ivy.to_numpy(c_true))


@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 3])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call_with_cont_input(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = ivy.Container(cont_key=tensor_fn(x0, 'float32', dev_str))
    in1 = ivy.Container(cont_key=tensor_fn(x1, 'float32', dev_str))

    # function
    def func(t0, t1):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call(func, [in0, in1], chunk_size, axis)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a.cont_key), ivy.to_numpy(a_true.cont_key))
    assert np.allclose(ivy.to_numpy(b.cont_key), ivy.to_numpy(b_true.cont_key))
    assert np.allclose(ivy.to_numpy(c.cont_key), ivy.to_numpy(c_true.cont_key))


@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 2])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call_across_gpus(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = tensor_fn(x0, 'float32', dev_str)
    in1 = tensor_fn(x1, 'float32', dev_str)

    # function
    # noinspection PyShadowingNames
    def func(t0, t1, dev_str=None):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    dev_str0 = dev_str
    if 'gpu' in dev_str:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
    else:
        dev_str1 = dev_str
    a, b, c = ivy.split_func_call_across_gpus(func, [in0, in1], [dev_str0, dev_str1], axis, concat_output=True)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a), ivy.to_numpy(a_true))
    assert np.allclose(ivy.to_numpy(b), ivy.to_numpy(b_true))
    assert np.allclose(ivy.to_numpy(c), ivy.to_numpy(c_true))


@pytest.mark.parametrize(
    "x0", [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
           [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
@pytest.mark.parametrize(
    "x1", [[[2, 4, 6], [8, 10, 12], [14, 16, 18]],
           [[18, 16, 14], [12, 10, 8], [6, 4, 2]]])
@pytest.mark.parametrize(
    "chunk_size", [1, 2])
@pytest.mark.parametrize(
    "axis", [0, 1])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split_func_call_across_gpus_with_cont_input(x0, x1, chunk_size, axis, tensor_fn, dev_str, call):

    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes, only integer number of sections input is supported.
        pytest.skip()

    # inputs
    in0 = ivy.Container(cont_key=tensor_fn(x0, 'float32', dev_str))
    in1 = ivy.Container(cont_key=tensor_fn(x1, 'float32', dev_str))

    # function
    # noinspection PyShadowingNames
    def func(t0, t1, dev_str=None):
        return t0 * t1, t0 - t1, t1 - t0

    # predictions
    a, b, c = ivy.split_func_call_across_gpus(func, [in0, in1], ["cpu:0", "cpu:0"], axis, concat_output=True)

    # true
    a_true, b_true, c_true = func(in0, in1)

    # value test
    assert np.allclose(ivy.to_numpy(a.cont_key), ivy.to_numpy(a_true.cont_key))
    assert np.allclose(ivy.to_numpy(b.cont_key), ivy.to_numpy(b_true.cont_key))
    assert np.allclose(ivy.to_numpy(c.cont_key), ivy.to_numpy(c_true.cont_key))


def test_cache_fn(dev_str, call):

    def func():
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn()
    ret0_again = cached_fn()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions each use the same global dict
    ret0 = ivy.cache_fn(func)()
    ret0_again = ivy.cache_fn(func)()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


def test_cache_fn_with_args(dev_str, call):

    def func(_):
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn(0)
    ret0_again = cached_fn(0)
    ret1 = cached_fn(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions each use the same global dict
    ret0 = ivy.cache_fn(func)(0)
    ret0_again = ivy.cache_fn(func)(0)
    ret1 = ivy.cache_fn(func)(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


def test_framework_setting_with_threading(dev_str, call):

    if call is helpers.np_call:
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def thread_fn():
        ivy.set_framework('numpy')
        x_ = np.array([0., 1., 2.])
        for _ in range(1000):
            try:
                ivy.reduce_mean(x_)
            except TypeError:
                return False
        ivy.unset_framework()
        return True

    # get original framework string and array
    fws = ivy.current_framework_str()
    x = ivy.array([0., 1., 2.])

    # start numpy loop thread
    thread = threading.Thread(target=thread_fn)
    thread.start()

    # start local original framework loop
    ivy.set_framework(fws)
    for _ in range(1000):
        ivy.reduce_mean(x)
    ivy.unset_framework()

    assert not thread.join()


def test_framework_setting_with_multiprocessing(dev_str, call):

    if call is helpers.np_call:
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def worker_fn(out_queue):
        ivy.set_framework('numpy')
        x_ = np.array([0., 1., 2.])
        for _ in range(1000):
            try:
                ivy.reduce_mean(x_)
            except TypeError:
                out_queue.put(False)
                return
        ivy.unset_framework()
        out_queue.put(True)

    # get original framework string and array
    fws = ivy.current_framework_str()
    x = ivy.array([0., 1., 2.])

    # start numpy loop thread
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=worker_fn, args=(output_queue,))
    worker.start()

    # start local original framework loop
    ivy.set_framework(fws)
    for _ in range(1000):
        ivy.reduce_mean(x)
    ivy.unset_framework()

    worker.join()
    assert output_queue.get_nowait()


def test_explicit_ivy_framework_handles(dev_str, call):

    if call is helpers.np_call:
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    # store original framework string and unset
    fw_str = ivy.current_framework_str()
    ivy.unset_framework()

    # set with explicit handle caught
    ivy_exp = ivy.get_framework(fw_str)
    assert ivy_exp.current_framework_str() == fw_str

    # assert backend implemented function is accessible
    assert 'array' in ivy_exp.__dict__
    assert callable(ivy_exp.array)

    # assert joint implemented function is also accessible
    assert 'cache_fn' in ivy_exp.__dict__
    assert callable(ivy_exp.cache_fn)

    # set global ivy to numpy
    ivy.set_framework('numpy')

    # assert the explicit handle is still unchanged
    assert ivy.current_framework_str() == 'numpy'
    assert ivy_exp.current_framework_str() == fw_str

    # unset global ivy from numpy
    ivy.unset_framework()


def test_class_ivy_handles(dev_str, call):

    if call is helpers.np_call:
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    class ArrayGen:

        def __init__(self, ivyh):
            self._ivy = ivyh

        def get_array(self):
            return self._ivy.array([0., 1., 2.])

    # create instance
    ag = ArrayGen(ivy.get_framework())

    # create array from array generator
    x = ag.get_array()

    # verify this is not a numpy array
    assert not isinstance(x, np.ndarray)

    # change global framework to numpy
    ivy.set_framework('numpy')

    # create another array from array generator
    x = ag.get_array()

    # verify this is not still a numpy array
    assert not isinstance(x, np.ndarray)


# einops_rearrange
@pytest.mark.parametrize(
    "x_n_pattern_n_newx", [([[0., 1., 2., 3.]], 'b n -> n b', [[0.], [1.], [2.], [3.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_rearrange(x_n_pattern_n_newx, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, pattern, new_x = x_n_pattern_n_newx
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.einops_rearrange(x, pattern)
    true_ret = einops.rearrange(ivy.to_native(x), pattern)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))
    # compilation test
    if call is helpers.torch_call:
        # torch jit cannot compile **args
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.einops_rearrange)


# einops_reduce
@pytest.mark.parametrize(
    "x_n_pattern_n_red_n_newx", [([[0., 1., 2., 3.]], 'b n -> b', 'mean', [1.5])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_reduce(x_n_pattern_n_red_n_newx, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, pattern, reduction, new_x = x_n_pattern_n_red_n_newx
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.einops_reduce(x, pattern, reduction)
    true_ret = einops.reduce(ivy.to_native(x), pattern, reduction)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))
    # compilation test
    if call is helpers.torch_call:
        # torch jit cannot compile **args
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.einops_reduce)


# einops_repeat
@pytest.mark.parametrize(
    "x_n_pattern_n_al_n_newx", [([[0., 1., 2., 3.]], 'b n -> b n c', {'c': 2},
                                 [[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_repeat(x_n_pattern_n_al_n_newx, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, pattern, axes_lengths, new_x = x_n_pattern_n_al_n_newx
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.einops_repeat(x, pattern, **axes_lengths)
    true_ret = einops.repeat(ivy.to_native(x), pattern, **axes_lengths)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))
    # compilation test
    if call is helpers.torch_call:
        # torch jit cannot compile **args
        pytest.skip()
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.einops_repeat)


# profiler
def test_profiler(dev_str, call):

    # with statement
    with ivy.Profiler('log'):
        a = ivy.ones([10])
        b = ivy.zeros([10])
        a + b
    if call is helpers.mx_call:
        time.sleep(1)  # required by MXNet for some reason

    # start and stop methods
    profiler = ivy.Profiler('log')
    profiler.start()
    a = ivy.ones([10])
    b = ivy.zeros([10])
    a + b
    profiler.stop()
    if call is helpers.mx_call:
        time.sleep(1)  # required by MXNet for some reason
