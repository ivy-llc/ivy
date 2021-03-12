"""
Collection of tests for templated general functions
"""

# global
import pytest
import ivy.numpy
import numpy as np
from numbers import Number
from operator import mul as _mul
from functools import reduce as _reduce

# local
import ivy
import ivy_tests.helpers as helpers

from collections.abc import Sequence


# Helpers #
# --------#

def _var_fn(a, b=None, c=None):
    return ivy.variable(ivy.array(a, b, c))


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


def np_scatter(indices, updates, shape, reduction='sum'):
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == 'sum':
        target = np.zeros(shape, dtype=updates.dtype)
        np.add.at(target, indices_tuple, updates)
    elif reduction == 'min':
        target = np.ones(shape, dtype=updates.dtype)*1e12
        np.minimum.at(target, indices_tuple, updates)
        target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        target = np.ones(shape, dtype=updates.dtype)*-1e12
        np.maximum.at(target, indices_tuple, updates)
        target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('Invalid reduction selected')
    return target


# Tests #
# ------#

# tensor
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_array(object_in, dtype_str, dev_str, call):
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    # smoke test
    ret = ivy.array(object_in, dtype_str, dev_str)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(call(ivy.array, object_in, dtype_str, dev_str), np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
    helpers.assert_compilable(ivy.array)


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
    helpers.assert_compilable(ivy.to_numpy)


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
    helpers.assert_compilable(ivy.to_list)


# shape
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_shape(object_in, dtype_str, as_tensor, tensor_fn, dev_str, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.shape(tensor_fn(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert isinstance(ret, ivy.Tensor)
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
    helpers.assert_compilable(ivy.shape)


# get_num_dims
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_get_num_dims(object_in, dtype_str, as_tensor, tensor_fn, dev_str, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.get_num_dims(tensor_fn(object_in, dtype_str, dev_str), as_tensor)
    # type test
    if as_tensor:
        assert isinstance(ret, ivy.Tensor)
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
    helpers.assert_compilable(ivy.shape)


# minimum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_minimum(xy, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number)) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype_str, dev_str)
    y = tensor_fn(xy[1], dtype_str, dev_str)
    ret = ivy.minimum(x, y)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.minimum, x, y), ivy.numpy.minimum(ivy.to_numpy(x), ivy.to_numpy(y)))
    # compilation test
    helpers.assert_compilable(ivy.minimum)


# maximum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_maximum(xy, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number)) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype_str, dev_str)
    y = tensor_fn(xy[1], dtype_str, dev_str)
    ret = ivy.maximum(x, y)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.maximum, x, y), ivy.numpy.maximum(ivy.to_numpy(x), ivy.to_numpy(y)))
    # compilation test
    helpers.assert_compilable(ivy.maximum)


# clip
@pytest.mark.parametrize(
    "x_min_n_max", [(-0.5, 0., 1.5), ([1.7], [0.5], [1.1]), ([[0.8, 2.2], [1.5, 0.2]], 0.2, 1.4),
                    ([[0.8, 2.2], [1.5, 0.2]], [[1., 1.], [1., 1.]], [[1.1, 2.], [1.1, 2.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_clip(x_min_n_max, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_min_n_max[0], Number) or isinstance(x_min_n_max[1], Number) or isinstance(x_min_n_max[2], Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_min_n_max[0], dtype_str, dev_str)
    min_val = tensor_fn(x_min_n_max[1], dtype_str, dev_str)
    max_val = tensor_fn(x_min_n_max[2], dtype_str, dev_str)
    if ((min_val.shape != [] and min_val.shape != [1]) or (max_val.shape != [] and max_val.shape != [1]))\
            and call in [helpers.torch_call, helpers.mx_call]:
        # pytorch and mxnet only support numbers or 0 or 1 dimensional arrays for min and max while performing clip
        pytest.skip()
    ret = ivy.clip(x, min_val, max_val)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    max_shape = max([x.shape, min_val.shape, max_val.shape], key=lambda x_: len(x_))
    assert ret.shape == max_shape
    # value test
    assert np.array_equal(call(ivy.clip, x, min_val, max_val),
                          ivy.numpy.clip(ivy.to_numpy(x), ivy.to_numpy(min_val), ivy.to_numpy(max_val)))
    # compilation test
    helpers.assert_compilable(ivy.clip)


# round
@pytest.mark.parametrize(
    "x_n_x_rounded", [(-0.51, -1), ([1.7], [2.]), ([[0.8, 2.2], [1.51, 0.2]], [[1., 2.], [2., 0.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_round(x_n_x_rounded, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_rounded[0], Number) or isinstance(x_n_x_rounded[1], Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_rounded[0], dtype_str, dev_str)
    ret = ivy.round(x)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.array_equal(call(ivy.round, x), np.array(x_n_x_rounded[1]))
    # compilation test
    helpers.assert_compilable(ivy.round)


# floormod
@pytest.mark.parametrize(
    "x_n_divisor_n_x_floormod", [(2.5, 2., 0.5), ([10.7], [5.], [0.7]),
                                 ([[0.8, 2.2], [1.7, 0.2]], [[0.3, 0.5], [0.4, 0.11]], [[0.2, 0.2], [0.1, 0.09]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_floormod(x_n_divisor_n_x_floormod, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_divisor_n_x_floormod[0], Number) or isinstance(x_n_divisor_n_x_floormod[1], Number) or
            isinstance(x_n_divisor_n_x_floormod[2], Number)) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_divisor_n_x_floormod[0], dtype_str, dev_str)
    divisor = ivy.array(x_n_divisor_n_x_floormod[1], dtype_str, dev_str)
    ret = ivy.floormod(x, divisor)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floormod, x, divisor), np.array(x_n_divisor_n_x_floormod[2]))
    # compilation test
    helpers.assert_compilable(ivy.floormod)


# floor
@pytest.mark.parametrize(
    "x_n_x_floored", [(2.5, 2.), ([10.7], [10.]), ([[3.8, 2.2], [1.7, 0.2]], [[3., 2.], [1., 0.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_floor(x_n_x_floored, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_floored[0], Number) or isinstance(x_n_x_floored[1], Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_floored[0], dtype_str, dev_str)
    ret = ivy.floor(x)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floor, x), np.array(x_n_x_floored[1]))
    # compilation test
    helpers.assert_compilable(ivy.floor)


# ceil
@pytest.mark.parametrize(
    "x_n_x_ceiled", [(2.5, 3.), ([10.7], [11.]), ([[3.8, 2.2], [1.7, 0.2]], [[4., 3.], [2., 1.]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_ceil(x_n_x_ceiled, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_ceiled[0], Number) or isinstance(x_n_x_ceiled[1], Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_ceiled[0], dtype_str, dev_str)
    ret = ivy.ceil(x)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.ceil, x), np.array(x_n_x_ceiled[1]))
    # compilation test
    helpers.assert_compilable(ivy.ceil)


# abs
@pytest.mark.parametrize(
    "x_n_x_absed", [(-2.5, 2.5), ([-10.7], [10.7]), ([[-3.8, 2.2], [1.7, -0.2]], [[3.8, 2.2], [1.7, 0.2]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_abs(x_n_x_absed, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_absed[0], Number) or isinstance(x_n_x_absed[1], Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_absed[0], dtype_str, dev_str)
    ret = ivy.abs(x)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.abs, x), np.array(x_n_x_absed[1]))
    # compilation test
    helpers.assert_compilable(ivy.abs)


# argmax
@pytest.mark.parametrize(
    "x_n_axis_x_argmax", [([-0.3, 0.1], None, [1]), ([[1.3, 2.6], [2.3, 2.5]], 0, [1, 0]),
                          ([[1.3, 2.6], [2.3, 2.5]], 1, [1, 1])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_argmax(x_n_axis_x_argmax, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = ivy.array(x_n_axis_x_argmax[0], dtype_str, dev_str)
    axis = x_n_axis_x_argmax[1]
    ret = ivy.argmax(x, axis)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert tuple(ret.shape) == (len(x.shape),)
    # value test
    assert np.allclose(call(ivy.argmax, x, axis), np.array(x_n_axis_x_argmax[2]))
    # compilation test
    helpers.assert_compilable(ivy.argmax)


# argmin
@pytest.mark.parametrize(
    "x_n_axis_x_argmin", [([-0.3, 0.1], None, [0]), ([[1.3, 2.6], [2.3, 2.5]], 0, [0, 1]),
                          ([[1.3, 2.6], [2.3, 2.5]], 1, [0, 0])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_argmin(x_n_axis_x_argmin, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x_n_axis_x_argmin[0], dtype_str, dev_str)
    axis = x_n_axis_x_argmin[1]
    ret = ivy.argmin(x, axis)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert tuple(ret.shape) == (len(x.shape),)
    # value test
    assert np.allclose(call(ivy.argmin, x, axis), np.array(x_n_axis_x_argmin[2]))
    # compilation test
    helpers.assert_compilable(ivy.argmin)


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
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert ivy.dtype_str(ret) == target_dtype_str
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    helpers.assert_compilable(ivy.cast)


# arange
@pytest.mark.parametrize(
    "stop_n_start_n_step", [[10, None, None], [10, 2, None], [10, 2, 2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_arange(stop_n_start_n_step, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    stop, start, step = stop_n_start_n_step
    if (isinstance(stop, Number) or isinstance(start, Number) or isinstance(step, Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    if tensor_fn == _var_fn and call is helpers.torch_call:
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
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == (int((ivy.to_list(stop) -
                              (ivy.to_list(start) if start else 0))/(ivy.to_list(step) if step else 1)),)
    # value test
    assert np.array_equal(call(ivy.arange, *args, dtype_str=dtype_str, dev_str=dev_str),
                          ivy.numpy.arange(*[ivy.to_numpy(arg) for arg in args], dtype_str=dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support Number type, or Union for Union[float, int] etc.
        return
    helpers.assert_compilable(ivy.arange)


# linspace
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_axis", [[1, 10, 100, None], [[[0., 1., 2.]], [[1., 2., 3.]], 150, -1],
                                  [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, -2]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_linspace(start_n_stop_n_num_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    start, stop, num, axis = start_n_stop_n_num_n_axis
    if (isinstance(start, Number) or isinstance(stop, Number))\
            and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype_str, dev_str)
    stop = tensor_fn(stop, dtype_str, dev_str)
    ret = ivy.linspace(start, stop, num, axis, dev_str=dev_str)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(call(ivy.linspace, start, stop, num, axis, dev_str=dev_str),
                       ivy.numpy.linspace(ivy.to_numpy(start), ivy.to_numpy(stop), num, axis))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    helpers.assert_compilable(ivy.linspace)


# concatenate
@pytest.mark.parametrize(
    "x1_n_x2_n_axis", [(1, 10, 0), ([[0., 1., 2.]], [[1., 2., 3.]], 0), ([[0., 1., 2.]], [[1., 2., 3.]], 1),
                       ([[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], -1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_concatenate(x1_n_x2_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x1, x2, axis = x1_n_x2_n_axis
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.concatenate((x1, x2), axis)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    axis_val = (axis % len(x1.shape) if (axis is not None and len(x1.shape) != 0) else len(x1.shape) - 1)
    if x1.shape == ():
        expected_shape = (2,)
    else:
        expected_shape = tuple([item * 2 if i == axis_val else item for i, item in enumerate(x1.shape)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.concatenate, [x1, x2], axis),
                       ivy.numpy.concatenate([ivy.to_numpy(x1), ivy.to_numpy(x2)], axis))
    # compilation test
    helpers.assert_compilable(ivy.concatenate)


# flip
@pytest.mark.parametrize(
    "x_n_axis_n_bs", [(1, 0, None), ([[0., 1., 2.]], None, (1, 3)), ([[0., 1., 2.]], 1, (1, 3)),
                       ([[[-0.1471, 0.4477, 0.2214]]], None, None)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_flip(x_n_axis_n_bs, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis, bs = x_n_axis_n_bs
    if isinstance(x, Number) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.flip(x, axis, bs)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.flip, x, axis, bs), ivy.numpy.flip(ivy.to_numpy(x), axis, bs))
    # compilation test
    helpers.assert_compilable(ivy.flip)


# stack
@pytest.mark.parametrize(
    "xs_n_axis", [((1, 0), -1), (([[0., 1., 2.]], [[3., 4., 5.]]), 0), (([[0., 1., 2.]], [[3., 4., 5.]]), 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_stack(xs_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    (x1, x2), axis = xs_n_axis
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = tensor_fn(x1, dtype_str, dev_str)
    x2 = tensor_fn(x2, dtype_str, dev_str)
    ret = ivy.stack((x1, x2), axis)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    axis_val = (axis % len(x1.shape) if (axis is not None and len(x1.shape) != 0) else len(x1.shape) - 1)
    if x1.shape == ():
        expected_shape = (2,)
    else:
        expected_shape = list(x1.shape)
        expected_shape.insert(axis_val, 2)
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.stack, (x1, x2), axis), ivy.numpy.stack((ivy.to_numpy(x1), ivy.to_numpy(x2)), axis))
    # compilation test
    helpers.assert_compilable(ivy.stack)


# unstack
@pytest.mark.parametrize(
    "x_n_axis", [(1, -1), ([[0., 1., 2.]], 0), ([[0., 1., 2.]], 1)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_unstack(x_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, axis = x_n_axis
    if isinstance(x, Number) and tensor_fn == _var_fn and call is helpers.mx_call:
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
    assert np.allclose(call(ivy.unstack, x, axis), ivy.numpy.unstack(ivy.to_numpy(x), axis))
    # compilation test
    helpers.assert_compilable(ivy.unstack)


# split
@pytest.mark.parametrize(
    "x_n_secs_n_axis", [(1, 1, -1), ([[0., 1., 2., 3.]], 2, 1), ([[0., 1., 2.], [3., 4., 5.]], 2, 0)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_split(x_n_secs_n_axis, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, secs, axis = x_n_secs_n_axis
    if isinstance(x, Number) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.split(x, secs, axis)
    # type test
    assert isinstance(ret, list)
    # cardinality test
    axis_val = (axis % len(x.shape) if (axis is not None and len(x.shape) != 0) else len(x.shape) - 1)
    if x.shape == ():
        expected_shape = ()
    else:
        expected_shape = tuple([int(item/secs) if i == axis_val else item for i, item in enumerate(x.shape)])
    assert ret[0].shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.split, x, secs, axis), ivy.numpy.split(ivy.to_numpy(x), secs, axis))
    # compilation test
    helpers.assert_compilable(ivy.split)


# tile
@pytest.mark.parametrize(
    "x_n_reps", [(1, [1]), (1, 2), (1, [2]), ([[0., 1., 2., 3.]], (2, 1))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, _var_fn])
def test_tile(x_n_reps, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, reps_raw = x_n_reps
    if isinstance(x, Number) and tensor_fn == _var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype_str, dev_str)
    ret_from_list = ivy.tile(x, reps_raw)
    reps = ivy.array(reps_raw, 'int32', dev_str)
    ret = ivy.tile(x, reps)
    # type test
    assert isinstance(ret, ivy.Tensor)
    # cardinality test
    if x.shape == ():
        expected_shape = tuple(reps_raw) if isinstance(reps_raw, list) else (reps_raw,)
    else:
        expected_shape = tuple([int(item * rep) for item, rep in zip(x.shape, reps_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.tile, x, reps), ivy.numpy.tile(ivy.to_numpy(x), ivy.to_numpy(reps)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return
    helpers.assert_compilable(ivy.tile)


def test_zero_pad(dev_str, call):
    assert np.array_equal(call(ivy.zero_pad, ivy.array([[0.]]), [[0, 1], [1, 2]], x_shape=[1, 1]),
                          np.pad(np.array([[0.]]), [[0, 1], [1, 2]]))
    assert np.array_equal(call(ivy.zero_pad, ivy.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]],
                               x_shape=[1, 1, 1]),
                          np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]]))
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.zero_pad)


def test_constant_pad(dev_str, call):
    assert np.array_equal(call(ivy.constant_pad, ivy.array([[0.]]), [[0, 1], [1, 2]], 2.,
                               x_shape=[1, 1]), np.pad(np.array([[0.]]), [[0, 1], [1, 2]], constant_values=2.))
    assert np.array_equal(call(ivy.constant_pad, ivy.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]],
                               3., x_shape=[1, 1, 1]),
                          np.pad(np.array([[[0.]]]), [[0, 0], [1, 1], [2, 3]], constant_values=3.))
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.constant_pad)


def test_swapaxes(dev_str, call):
    assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]]), 1, 0),
                          np.swapaxes(np.array([[0., 0.]]), 1, 0))
    assert np.array_equal(call(ivy.swapaxes, ivy.array([[0., 0.]]), -1, -2),
                          np.swapaxes(np.array([[0., 0.]]), -1, -2))
    helpers.assert_compilable(ivy.swapaxes)


def test_transpose(dev_str, call):
    assert np.array_equal(call(ivy.transpose, ivy.array([[0., 0.]]), [1, 0]),
                          np.transpose(np.array([[0., 0.]]), [1, 0]))
    assert np.array_equal(call(ivy.transpose, ivy.array([[[0., 0.]]]), [2, 0, 1]),
                          np.transpose(np.array([[[0., 0.]]]), [2, 0, 1]))
    helpers.assert_compilable(ivy.transpose)


def test_expand_dims(dev_str, call):
    assert np.array_equal(call(ivy.expand_dims, ivy.array([[0., 0.]]), 0),
                          np.expand_dims(np.array([[0., 0.]]), 0))
    assert np.array_equal(call(ivy.expand_dims, ivy.array([[[0., 0.]]]), -1),
                          np.expand_dims(np.array([[[0., 0.]]]), -1))
    helpers.assert_compilable(ivy.expand_dims)


def test_where(dev_str, call):
    assert np.array_equal(call(ivy.where, ivy.array([[0., 1.]]) > 0,
                               ivy.array([[1., 1.]]), ivy.array([[2., 2.]]),
                               condition_shape=[1, 2], x_shape=[1, 2]),
                          np.where(np.array([[0., 1.]]) > 0, np.array([[0., 1.]]), np.array([[2., 2.]])))
    assert np.array_equal(call(ivy.where, ivy.array([[[1., 0.]]]) > 0,
                               ivy.array([[[1., 1.]]]), ivy.array([[[2., 2.]]]),
                               condition_shape=[1, 1, 2], x_shape=[1, 1, 2]),
                          np.where(np.array([[[1., 0.]]]) > 0, np.array([[[1., 1.]]]), np.array([[[2., 2.]]])))
    helpers.assert_compilable(ivy.where)


def test_indices_where(dev_str, call):
    assert np.array_equal(call(ivy.indices_where, ivy.array([[False, True],
                                                              [True, False],
                                                              [True, True]])),
                          np.array([[0, 1], [1, 0], [2, 0], [2, 1]]))
    assert np.array_equal(call(ivy.indices_where, ivy.array([[[False, True],
                                                               [True, False],
                                                               [True, True]]])),
                          np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1]]))
    helpers.assert_compilable(ivy.indices_where)


def test_reshape(dev_str, call):
    assert np.array_equal(call(ivy.reshape, ivy.array([[0., 1.]]), (-1,)),
                          np.reshape(np.array([[0., 1.]]), (-1,)))
    assert np.array_equal(call(ivy.reshape, ivy.array([[[1., 0.]]]), (1, 2)),
                          np.reshape(np.array([[[1., 0.]]]), (1, 2)))
    helpers.assert_compilable(ivy.reshape)


def test_squeeze(dev_str, call):
    assert np.array_equal(call(ivy.squeeze, ivy.array([[0., 1.]])),
                          np.squeeze(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.squeeze, ivy.array([[[1., 0.]]]), 1),
                          np.squeeze(np.array([[[1., 0.]]]), 1))
    helpers.assert_compilable(ivy.squeeze)


def test_zeros(dev_str, call):
    assert np.array_equal(call(ivy.zeros, (1, 2)), np.zeros((1, 2)))
    assert np.array_equal(call(ivy.zeros, (1, 2), 'int64'), np.zeros((1, 2), np.int64))
    assert np.array_equal(call(ivy.zeros, (1, 2, 3)), np.zeros((1, 2, 3)))
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.zeros)


def test_zeros_like(dev_str, call):
    assert np.array_equal(call(ivy.zeros_like, ivy.array([[0., 1.]])),
                          np.zeros_like(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.zeros_like, ivy.array([[[1., 0.]]])),
                          np.zeros_like(np.array([[[1., 0.]]])))
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.zeros_like)


def test_ones(dev_str, call):
    assert np.array_equal(call(ivy.ones, (1, 2)), np.ones((1, 2)))
    assert np.array_equal(call(ivy.ones, (1, 2), 'int64'), np.ones((1, 2), np.int64))
    assert np.array_equal(call(ivy.ones, (1, 2, 3)), np.ones((1, 2, 3)))
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.ones)


def test_ones_like(dev_str, call):
    assert np.array_equal(call(ivy.ones_like, ivy.array([[0., 1.]])),
                          np.ones_like(np.array([[0., 1.]])))
    assert np.array_equal(call(ivy.ones_like, ivy.array([[[1., 0.]]])),
                          np.ones_like(np.array([[[1., 0.]]])))
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.ones_like)


def test_one_hot(dev_str, call):
    np_one_hot = helpers._ivy_np.one_hot(np.array([0, 1, 2]), 3)
    assert np.array_equal(call(ivy.one_hot, ivy.array([0, 1, 2]), 3), np_one_hot)
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.one_hot)


def test_cross(dev_str, call):
    assert np.array_equal(call(ivy.cross, ivy.array([0., 0., 0.]),
                               ivy.array([0., 0., 0.])),
                          np.cross(np.array([0., 0., 0.]), np.array([0., 0., 0.])))
    assert np.array_equal(call(ivy.cross, ivy.array([[0., 0., 0.]]),
                               ivy.array([[0., 0., 0.]])),
                          np.cross(np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])))
    helpers.assert_compilable(ivy.cross)


def test_matmul(dev_str, call):
    assert np.array_equal(call(ivy.matmul, ivy.array([[1., 0.], [0., 1.]]),
                               ivy.array([[1., 0.], [0., 1.]]), batch_shape=[]),
                          np.matmul(np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.], [0., 1.]])))
    assert np.array_equal(call(ivy.matmul, ivy.array([[[[1., 0.], [0., 1.]]]]),
                               ivy.array([[[[1., 0.], [0., 1.]]]]), batch_shape=[1, 1]),
                          np.matmul(np.array([[[[1., 0.], [0., 1.]]]]), np.array([[[[1., 0.], [0., 1.]]]])))
    helpers.assert_compilable(ivy.matmul)


def test_cumsum(dev_str, call):
    assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2., 3.]]), 1),
                          np.array([[0., 1., 3., 6.]]))
    assert np.array_equal(call(ivy.cumsum, ivy.array([[0., 1., 2.], [0., 1., 2.]]), 0),
                          np.array([[0., 1., 2.], [0., 2., 4.]]))
    helpers.assert_compilable(ivy.cumsum)


def test_identity(dev_str, call):
    assert np.array_equal(call(ivy.identity, 1), np.identity(1))
    assert np.array_equal(call(ivy.identity, 2, 'int64'), np.identity(2, np.int64))
    call(ivy.identity, 2, 'int64', (1, 2))
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.identity)


def test_scatter_flat_sum(dev_str, call):
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2]),
                               ivy.array([1, 2, 3, 4]), 8),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.mx_call]:
        # mxnet scatter does not support sum reduction
        return
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0]),
                               ivy.array([1, 2, 3, 4, 5]), 8),
                          np.array([6, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_flat_min(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter flat
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2]),
                               ivy.array([1, 2, 3, 4]), 8, 'min'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0]),
                               ivy.array([1, 2, 3, 4, 5]), 8, 'min'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_flat_max(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter flat
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2]),
                               ivy.array([1, 2, 3, 4]), 8, 'max'),
                          np.array([1, 3, 4, 0, 2, 0, 0, 0]))
    assert np.array_equal(call(ivy.scatter_flat, ivy.array([0, 4, 1, 2, 0]),
                               ivy.array([1, 2, 3, 4, 5]), 8, 'max'),
                          np.array([5, 3, 4, 0, 2, 0, 0, 0]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_flat)


def test_scatter_sum_nd(dev_str, call):
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]]),
                               ivy.array([9, 10, 11, 12]), [8], 2),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]),
                                             [8]))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]]),
                               ivy.array([1]), [3, 3, 3], 2),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]),
                                             [3, 3, 3]))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]]),
                               ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 2),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4]))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_scatter_min_nd(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support min reduction for scatter nd
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]]),
                               ivy.array([9, 10, 11, 12]), [8], 'min'),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'min'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]]),
                               ivy.array([1]), [3, 3, 3], 'min'),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'min'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]]),
                               ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'min'))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_scatter_max_nd(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support max reduction for scatter nd
        pytest.skip()
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[4], [3], [1], [7]]),
                               ivy.array([9, 10, 11, 12]), [8], 'max'),
                          np_scatter(np.array([[4], [3], [1], [7]]), np.array([9, 10, 11, 12]), [8], 'max'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0, 1, 2]]),
                               ivy.array([1]), [3, 3, 3], 'max'),
                          np_scatter(np.array([[0, 1, 2]]), np.array([1]), [3, 3, 3], 'max'))
    assert np.array_equal(call(ivy.scatter_nd, ivy.array([[0], [2]]),
                               ivy.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                            [7, 7, 7, 7], [8, 8, 8, 8]],
                                           [[5, 5, 5, 5], [6, 6, 6, 6],
                                               [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'),
                          np_scatter(np.array([[0], [2]]),
                                             np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]],
                                                       [[5, 5, 5, 5], [6, 6, 6, 6],
                                                        [7, 7, 7, 7], [8, 8, 8, 8]]]), [4, 4, 4], 'max'))
    if call in [helpers.torch_call]:
        # global torch_scatter var not supported when scripting
        return
    helpers.assert_compilable(ivy.scatter_nd)


def test_gather_flat(dev_str, call):
    assert np.allclose(call(ivy.gather_flat, ivy.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                            ivy.array([0, 4, 7])), np.array([9, 5, 2]), atol=1e-6)
    if call in [helpers.torch_call]:
        # pytorch scripting cannot assign a torch.device value with a string
        return
    helpers.assert_compilable(ivy.gather_flat)


def test_gather_nd(dev_str, call):
    assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.array([[0, 1], [1, 0]]), indices_shape=[2, 2]),
                       np.array([[2.0, 3.0], [0.1, 1.1]]), atol=1e-6)
    assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.array([[[0, 1]], [[1, 0]]]), indices_shape=[2, 1, 2]),
                       np.array([[[2.0, 3.0]], [[0.1, 1.1]]]), atol=1e-6)
    assert np.allclose(call(ivy.gather_nd, ivy.array([[[0.0, 1.0], [2.0, 3.0]],
                                                       [[0.1, 1.1], [2.1, 3.1]]]),
                            ivy.array([[[0, 1, 0]], [[1, 0, 1]]]),
                            indices_shape=[2, 1, 3]), np.array([[2.0], [1.1]]), atol=1e-6)
    if call in [helpers.torch_call]:
        # torch scripting does not support builtins
        return
    helpers.assert_compilable(ivy.gather_nd)


def test_dev(dev_str, call):
    assert ivy.dev(ivy.array([1.]))
    helpers.assert_compilable(ivy.dev)


def test_dev_to_str(dev_str, call):
    assert 'cpu' in ivy.dev_to_str(ivy.dev(ivy.array([0.]))).lower()
    helpers.assert_compilable(ivy.dev_to_str)


def test_dev_str(dev_str, call):
    assert 'cpu' in ivy.dev_str(ivy.array([0.])).lower()
    helpers.assert_compilable(ivy.dev_str)


def test_dtype(dev_str, call):
    assert ivy.dtype(ivy.array([0.])) == ivy.array([0.]).dtype
    helpers.assert_compilable(ivy.dtype)


def test_dtype_to_str(dev_str, call):
    assert ivy.dtype_to_str(ivy.array([0.], dtype_str='float32').dtype) == 'float32'
    helpers.assert_compilable(ivy.dtype_to_str)


def test_dtype_str(dev_str, call):
    assert ivy.dtype_str(ivy.array([0.], dtype_str='float32')) == 'float32'
    helpers.assert_compilable(ivy.dtype_str)


def test_compile_fn(dev_str, call):
    some_fn = lambda x: x**2
    example_inputs = ivy.array([2.])
    new_fn = ivy.compile_fn(some_fn, False, example_inputs)
    assert np.allclose(call(new_fn, example_inputs), np.array([4.]))
