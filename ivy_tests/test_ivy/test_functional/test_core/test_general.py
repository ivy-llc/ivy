"""
Collection of tests for unified general functions
"""

# global
import os
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
import ivy.functional.backends.numpy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet
import ivy_tests.test_ivy.helpers as helpers


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
    "fw_str", ['numpy', 'jax', 'torch', 'mxnet'])
def test_set_framework(fw_str, dev, call):
    ivy.set_framework(fw_str)
    ivy.unset_framework()


# use_framework
def test_use_within_use_framework(dev, call):
    with ivy.functional.backends.numpy.use:
        pass
    with ivy.functional.backends.jax.use:
        pass
    with ivy.functional.backends.tensorflow.use:
        pass
    with ivy.functional.backends.torch.use:
        pass
    with ivy.functional.backends.mxnet.use:
        pass


@pytest.mark.parametrize(
    "allow_duplicates", [True, False])
def test_match_kwargs(allow_duplicates):

    def func_a(a, b, c=2):
        pass

    func_b = lambda a, d, e=5: None

    class ClassA:
        def __init__(self, c, f, g=3):
            pass

    kwargs = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
    kwfa, kwfb, kwca = ivy.match_kwargs(kwargs, func_a, func_b, ClassA, allow_duplicates=allow_duplicates)
    if allow_duplicates:
        assert kwfa == {'a': 0, 'b': 1, 'c': 2}
        assert kwfb == {'a': 0, 'd': 3, 'e': 4}
        assert kwca == {'c': 2, 'f': 5, 'g': 6}
    else:
        assert kwfa == {'a': 0, 'b': 1, 'c': 2}
        assert kwfb == {'d': 3, 'e': 4}
        assert kwca == {'f': 5, 'g': 6}


# def test_get_referrers_recursive(dev, call):
#
#     class SomeClass:
#         def __init__(self):
#             self.x = [1, 2]
#             self.y = [self.x]
#
#     some_obj = SomeClass()
#     refs = ivy.get_referrers_recursive(some_obj.x)
#     ref_keys = refs.keys()
#     assert len(ref_keys) == 3
#     assert 'repr' in ref_keys
#     assert refs['repr'] == '[1,2]'
#     y_id = str(id(some_obj.y))
#     y_refs = refs[y_id]
#     assert y_refs['repr'] == '[[1,2]]'
#     some_obj_dict_id = str(id(some_obj.__dict__))
#     assert y_refs[some_obj_dict_id] == 'tracked'
#     dict_refs = refs[some_obj_dict_id]
#     assert dict_refs['repr'] == "{'x':[1,2],'y':[[1,2]]}"
#     some_obj_id = str(id(some_obj))
#     some_obj_refs = dict_refs[some_obj_id]
#     assert some_obj_refs['repr'] == str(some_obj).replace(' ', '')
#     assert len(some_obj_refs) == 1


# array
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "from_numpy", [True, False])
def test_array(object_in, dtype, from_numpy, dev, call):
    if call in [helpers.mx_call] and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.array(object_in, dtype, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(call(ivy.array, object_in, dtype, dev), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return


# copy array
@pytest.mark.parametrize(
    "x", [[0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_copy_array(x, dtype, dev, call):
    if call in [helpers.mx_call] and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    # smoke test
    x = ivy.array(x, dtype, dev)
    ret = ivy.copy_array(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(x))
    assert id(x) != id(ret)
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return


# array_equal
@pytest.mark.parametrize(
    "x0_n_x1_n_res", [([0.], [0.], True), ([0.], [1.], False),
                      ([[0.], [1.]], [[0.], [1.]], True),
                      ([[0.], [1.]], [[1.], [2.]], False)])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
def test_array_equal(x0_n_x1_n_res, dtype, dev, call):
    if call in [helpers.mx_call] and dtype in ['int16', 'bool']:
        # mxnet does not support int16, and does not support bool for broadcast_equal method used
        pytest.skip()
    x0, x1, true_res = x0_n_x1_n_res
    # smoke test
    x0 = ivy.array(x0, dtype, dev)
    x1 = ivy.array(x1, dtype, dev)
    res = ivy.array_equal(x0, x1)
    # type test
    assert ivy.is_array(x0)
    assert ivy.is_array(x1)
    assert isinstance(res, bool) or ivy.is_array(res)
    # value test
    assert res == true_res


# arrays_equal
@pytest.mark.parametrize(
    "xs_n_res", [([[[0.], [1.]], [[0.], [1.]], [[1.], [2.]]], False)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
def test_arrays_equal(xs_n_res, dtype, dev, call):
    xs, true_res = xs_n_res
    # smoke test
    x0 = ivy.array(xs[0], dtype, dev)
    x1 = ivy.array(xs[1], dtype, dev)
    x2 = ivy.array(xs[2], dtype, dev)
    res = ivy.arrays_equal([x0, x1, x2])
    # type test
    assert ivy.is_array(x0)
    assert ivy.is_array(x1)
    assert ivy.is_array(x2)
    assert isinstance(res, bool) or ivy.is_array(res)
    # value test
    assert res == true_res


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
def test_equal(x0_n_x1_n_x2_em_n_res, to_array, dev, call):
    x0, x1, x2, equality_matrix, true_res = x0_n_x1_n_x2_em_n_res
    # smoke test
    if isinstance(x0, list) and to_array:
        x0 = ivy.array(x0, dev=dev)
        x1 = ivy.array(x1, dev=dev)
        x2 = ivy.array(x2, dev=dev)
    res = ivy.all_equal(x0, x1, x2, equality_matrix=equality_matrix)
    # value test
    if equality_matrix:
        assert np.array_equal(ivy.to_numpy(res), np.array(true_res))
    else:
        assert res == true_res
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support variable number of input arguments
        return


# to_numpy
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_numpy(object_in, dtype, tensor_fn, dev, call):
    if call in [helpers.mx_call] and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_numpy() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_numpy(tensor_fn(object_in, dtype, dev))
    # type test
    assert isinstance(ret, np.ndarray)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(ivy.to_numpy(tensor_fn(object_in, dtype, dev)), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support numpy conversion
        return


# to_scalar
@pytest.mark.parametrize(
    "object_in", [[0.], [[[1]]], [True], [[1.]]])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_scalar(object_in, dtype, tensor_fn, dev, call):
    if call in [helpers.mx_call] and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_scalar() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_scalar(tensor_fn(object_in, dtype, dev))
    true_val = ivy.to_numpy(ivy.array(object_in, dtype=dtype)).item()
    # type test
    assert isinstance(ret, type(true_val))
    # value test
    assert ivy.to_scalar(tensor_fn(object_in, dtype, dev)) == true_val
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support scalar conversion
        return


# to_list
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", [None, 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_to_list(object_in, dtype, tensor_fn, dev, call):
    if call in [helpers.mx_call] and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    # smoke test
    ret = ivy.to_list(tensor_fn(object_in, dtype, dev))
    # type test
    assert isinstance(ret, list)
    # cardinality test
    assert _get_shape_of_list(ret) == _get_shape_of_list(object_in)
    # value test
    assert np.allclose(np.asarray(ivy.to_list(tensor_fn(object_in, dtype, dev))),
                       np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support list conversion
        return


# shape
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_shape(object_in, dtype, as_tensor, tensor_fn, dev, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.shape(tensor_fn(object_in, dtype, dev), as_tensor)
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


# get_num_dims
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "as_tensor", [None, True, False])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_get_num_dims(object_in, dtype, as_tensor, tensor_fn, dev, call):
    # smoke test
    if len(object_in) == 0 and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ret = ivy.get_num_dims(tensor_fn(object_in, dtype, dev), as_tensor)
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


# minimum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_minimum(xy, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype, dev)
    y = tensor_fn(xy[1], dtype, dev)
    ret = ivy.minimum(x, y)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.minimum, x, y), np.asarray(ivy.functional.backends.numpy.minimum(ivy.to_numpy(x), ivy.to_numpy(y))))


# maximum
@pytest.mark.parametrize(
    "xy", [([0.7], [0.5]), ([0.7], 0.5), (0.5, [0.7]), ([[0.8, 1.2], [1.5, 0.2]], [0., 1.])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_maximum(xy, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(xy[0], Number) or isinstance(xy[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(xy[0], dtype, dev)
    y = tensor_fn(xy[1], dtype, dev)
    ret = ivy.maximum(x, y)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if len(x.shape) > len(y.shape):
        assert ret.shape == x.shape
    else:
        assert ret.shape == y.shape
    # value test
    assert np.array_equal(call(ivy.maximum, x, y), np.asarray(ivy.functional.backends.numpy.maximum(ivy.to_numpy(x), ivy.to_numpy(y))))


# clip
@pytest.mark.parametrize(
    "x_min_n_max", [(-0.5, 0., 1.5), ([1.7], [0.5], [1.1]), ([[0.8, 2.2], [1.5, 0.2]], 0.2, 1.4),
                    ([[0.8, 2.2], [1.5, 0.2]], [[1., 1.], [1., 1.]], [[1.1, 2.], [1.1, 2.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_clip(x_min_n_max, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_min_n_max[0], Number) or isinstance(x_min_n_max[1], Number) or isinstance(x_min_n_max[2], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_min_n_max[0], dtype, dev)
    min_val = tensor_fn(x_min_n_max[1], dtype, dev)
    max_val = tensor_fn(x_min_n_max[2], dtype, dev)
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
                          np.asarray(ivy.functional.backends.numpy.clip(ivy.to_numpy(x), ivy.to_numpy(min_val), ivy.to_numpy(max_val))))


# clip_vector_norm
# @pytest.mark.parametrize(
#     "x_max_norm_n_p_val_clipped",
#     [(-0.5, 0.4, 2., -0.4), ([1.7], 1.5, 3., [1.5]),
#      ([[0.8, 2.2], [1.5, 0.2]], 4., 1., [[0.6808511, 1.8723406], [1.2765958, 0.17021278]]),
#      ([[0.8, 2.2], [1.5, 0.2]], 2.5, 2., [[0.71749604, 1.9731141], [1.345305, 0.17937401]])])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_clip_vector_norm(x_max_norm_n_p_val_clipped, dtype, tensor_fn, dev, call):
#     # smoke test
#     if call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x_max_norm_n_p_val_clipped[0], dtype, dev)
#     max_norm = x_max_norm_n_p_val_clipped[1]
#     p_val = x_max_norm_n_p_val_clipped[2]
#     clipped = x_max_norm_n_p_val_clipped[3]
#     ret = ivy.clip_vector_norm(x, max_norm, p_val)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == (x.shape if len(x.shape) else (1,))
#     # value test
#     assert np.allclose(call(ivy.clip_vector_norm, x, max_norm, p_val), np.array(clipped))
#     # compilation test
#     if call is helpers.torch_call:
#         # pytorch jit cannot compile global variables, in this case MIN_DENOMINATOR
#         return


# round
@pytest.mark.parametrize(
    "x_n_x_rounded", [(-0.51, -1), ([1.7], [2.]), ([[0.8, 2.2], [1.51, 0.2]], [[1., 2.], [2., 0.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_round(x_n_x_rounded, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_n_x_rounded[0], Number) or isinstance(x_n_x_rounded[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_rounded[0], dtype, dev)
    ret = ivy.round(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.array_equal(call(ivy.round, x), np.array(x_n_x_rounded[1]))


# floormod
@pytest.mark.parametrize(
    "x_n_divisor_n_x_floormod", [(2.5, 2., 0.5), ([10.7], [5.], [0.7]),
                                 ([[0.8, 2.2], [1.7, 0.2]], [[0.3, 0.5], [0.4, 0.11]], [[0.2, 0.2], [0.1, 0.09]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_floormod(x_n_divisor_n_x_floormod, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_n_divisor_n_x_floormod[0], Number) or isinstance(x_n_divisor_n_x_floormod[1], Number) or
            isinstance(x_n_divisor_n_x_floormod[2], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_divisor_n_x_floormod[0], dtype, dev)
    divisor = ivy.array(x_n_divisor_n_x_floormod[1], dtype, dev)
    ret = ivy.floormod(x, divisor)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floormod, x, divisor), np.array(x_n_divisor_n_x_floormod[2]))


# floor
@pytest.mark.parametrize(
    "x_n_x_floored", [(2.5, 2.), ([10.7], [10.]), ([[3.8, 2.2], [1.7, 0.2]], [[3., 2.], [1., 0.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_floor(x_n_x_floored, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_n_x_floored[0], Number) or isinstance(x_n_x_floored[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_floored[0], dtype, dev)
    ret = ivy.floor(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.floor, x), np.array(x_n_x_floored[1]))


# ceil
@pytest.mark.parametrize(
    "x_n_x_ceiled", [(2.5, 3.), ([10.7], [11.]), ([[3.8, 2.2], [1.7, 0.2]], [[4., 3.], [2., 1.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_ceil(x_n_x_ceiled, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_n_x_ceiled[0], Number) or isinstance(x_n_x_ceiled[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_ceiled[0], dtype, dev)
    ret = ivy.ceil(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.ceil, x), np.array(x_n_x_ceiled[1]))


# abs
@pytest.mark.parametrize(
    "x_n_x_absed", [(-2.5, 2.5), ([-10.7], [10.7]), ([[-3.8, 2.2], [1.7, -0.2]], [[3.8, 2.2], [1.7, 0.2]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_abs(x_n_x_absed, dtype, tensor_fn, dev, call):
    # smoke test
    if (isinstance(x_n_x_absed[0], Number) or isinstance(x_n_x_absed[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_absed[0], dtype, dev)
    ret = ivy.abs(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.abs, x), np.array(x_n_x_absed[1]))


# argmax
# @pytest.mark.parametrize(
#     "x_n_axis_x_argmax", [([-0.3, 0.1], None, [1]), ([[1.3, 2.6], [2.3, 2.5]], 0, [1, 0]),
#                           ([[1.3, 2.6], [2.3, 2.5]], 1, [1, 1])])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_argmax(x_n_axis_x_argmax, dtype, tensor_fn, dev, call):
#     # smoke test
#     x = ivy.array(x_n_axis_x_argmax[0], dtype, dev)
#     axis = x_n_axis_x_argmax[1]
#     ret = ivy.argmax(x, axis)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert tuple(ret.shape) == (len(x.shape),)
#     # value test
#     assert np.allclose(call(ivy.argmax, x, axis), np.array(x_n_axis_x_argmax[2]))


# argmin
@pytest.mark.parametrize(
    "x_n_axis_x_argmin", [([-0.3, 0.1], None, [0]), ([[1.3, 2.6], [2.3, 2.5]], 0, [0, 1]),
                          ([[1.3, 2.6], [2.3, 2.5]], 1, [0, 0])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_argmin(x_n_axis_x_argmin, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x_n_axis_x_argmin[0], dtype, dev)
    axis = x_n_axis_x_argmin[1]
    ret = ivy.argmin(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert tuple(ret.shape) == (len(x.shape),)
    # value test
    assert np.allclose(call(ivy.argmin, x, axis), np.array(x_n_axis_x_argmin[2]))


# argsort
# @pytest.mark.parametrize(
#     "x_n_axis_x_argsort", [([1, 10, 26.9, 2.8, 166.32, 62.3], -1, [0, 3, 1, 2, 5, 4])])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_argsort(x_n_axis_x_argsort, dtype, tensor_fn, dev, call):
#     # smoke test
#     x = tensor_fn(x_n_axis_x_argsort[0], dtype, dev)
#     axis = x_n_axis_x_argsort[1]
#     ret = ivy.argsort(x, axis)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert tuple(ret.shape) == (6,)
#     # value test
#     assert np.allclose(call(ivy.argsort, x, axis), np.array(x_n_axis_x_argsort[2]))


# arange
@pytest.mark.parametrize(
    "stop_n_start_n_step", [[10, None, None], [10, 2, None], [10, 2, 2]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_arange(stop_n_start_n_step, dtype, tensor_fn, dev, call):
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
        stop = tensor_fn(stop, dtype, dev)
        args.append(stop)
    if start:
        start = tensor_fn(start, dtype, dev)
        args.append(start)
    if step:
        step = tensor_fn(step, dtype, dev)
        args.append(step)
    ret = ivy.arange(*args, dtype=dtype, dev=dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (int((ivy.to_list(stop) -
                              (ivy.to_list(start) if start else 0))/(ivy.to_list(step) if step else 1)),)
    # value test
    assert np.array_equal(call(ivy.arange, *args, dtype=dtype, dev=dev),
                          np.asarray(ivy.functional.backends.numpy.arange(*[ivy.to_numpy(arg) for arg in args], dtype=dtype)))


# linspace
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_axis", [[1, 10, 100, None], [[[0., 1., 2.]], [[1., 2., 3.]], 150, -1],
                                  [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, -2]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linspace(start_n_stop_n_num_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    start, stop, num, axis = start_n_stop_n_num_n_axis
    if (isinstance(start, Number) or isinstance(stop, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype, dev)
    stop = tensor_fn(stop, dtype, dev)
    ret = ivy.linspace(start, stop, num, axis, dev=dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(call(ivy.linspace, start, stop, num, axis, dev=dev),
                       np.asarray(ivy.functional.backends.numpy.linspace(ivy.to_numpy(start), ivy.to_numpy(stop), num, axis)))


# logspace
@pytest.mark.parametrize(
    "start_n_stop_n_num_n_base_n_axis", [[1, 10, 100, 10., None], [[[0., 1., 2.]], [[1., 2., 3.]], 150, 2., -1],
                                         [[[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], 6, 5., -2]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logspace(start_n_stop_n_num_n_base_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    start, stop, num, base, axis = start_n_stop_n_num_n_base_n_axis
    if (isinstance(start, Number) or isinstance(stop, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    start = tensor_fn(start, dtype, dev)
    stop = tensor_fn(stop, dtype, dev)
    ret = ivy.logspace(start, stop, num, base, axis, dev=dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    target_shape = list(start.shape)
    target_shape.insert(axis + 1 if (axis and axis != -1) else len(target_shape), num)
    assert ret.shape == tuple(target_shape)
    # value test
    assert np.allclose(call(ivy.logspace, start, stop, num, base, axis, dev=dev),
                       ivy.functional.backends.numpy.logspace(ivy.to_numpy(start), ivy.to_numpy(stop), num, base, axis))


# concatenate
@pytest.mark.parametrize(
    "x1_n_x2_n_axis", [(1, 10, 0), ([[0., 1., 2.]], [[1., 2., 3.]], 0), ([[0., 1., 2.]], [[1., 2., 3.]], 1),
                       ([[[-0.1471, 0.4477, 0.2214]]], [[[-0.3048, 0.3308, 0.2721]]], -1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_concatenate(x1_n_x2_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2, axis = x1_n_x2_n_axis
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = tensor_fn(x1, dtype, dev)
    x2 = tensor_fn(x2, dtype, dev)
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
                       np.asarray(ivy.functional.backends.numpy.concatenate([ivy.to_numpy(x1), ivy.to_numpy(x2)], axis)))


# flip
# @pytest.mark.parametrize(
#     "x_n_axis_n_bs", [(1, 0, None), ([[0., 1., 2.]], None, (1, 3)), ([[0., 1., 2.]], 1, (1, 3)),
#                        ([[[-0.1471, 0.4477, 0.2214]]], None, None)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_flip(x_n_axis_n_bs, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, axis, bs = x_n_axis_n_bs
#     if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x, dtype, dev)
#     ret = ivy.flip(x, axis, bs)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == x.shape
#     # value test
#     assert np.allclose(call(ivy.flip, x, axis, bs), np.asarray(ivy.functional.backends.numpy.flip(ivy.to_numpy(x), axis, bs)))


# stack
# @pytest.mark.parametrize(
#     "xs_n_axis", [((1, 0), -1), (([[0., 1., 2.]], [[3., 4., 5.]]), 0), (([[0., 1., 2.]], [[3., 4., 5.]]), 1)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_stack(xs_n_axis, dtype, tensor_fn, dev, call):
#     # smoke test
#     (x1, x2), axis = xs_n_axis
#     if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x1 = tensor_fn(x1, dtype, dev)
#     x2 = tensor_fn(x2, dtype, dev)
#     ret = ivy.stack((x1, x2), axis)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     axis_val = (axis % len(x1.shape) if (axis is not None and len(x1.shape) != 0) else len(x1.shape) - 1)
#     if x1.shape == ():
#         expected_shape = (2,)
#     else:
#         expected_shape = list(x1.shape)
#         expected_shape.insert(axis_val, 2)
#     assert ret.shape == tuple(expected_shape)
#     # value test
#     assert np.allclose(call(ivy.stack, (x1, x2), axis),
#                        np.asarray(ivy.functional.backends.numpy.stack((ivy.to_numpy(x1), ivy.to_numpy(x2)), axis)))


# unstack
@pytest.mark.parametrize(
    "x_n_axis", [(1, -1), ([[0., 1., 2.]], 0), ([[0., 1., 2.]], 1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_unstack(x_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    x, axis = x_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
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
    assert np.allclose(call(ivy.unstack, x, axis), np.asarray(ivy.functional.backends.numpy.unstack(ivy.to_numpy(x), axis)))


# split
@pytest.mark.parametrize(
    "x_n_noss_n_axis_n_wr", [(1, 1, -1, False),
                             ([[0., 1., 2., 3.]], 2, 1, False),
                             ([[0., 1., 2.], [3., 4., 5.]], 2, 0, False),
                             ([[0., 1., 2.], [3., 4., 5.]], 2, 1, True),
                             ([[0., 1., 2.], [3., 4., 5.]], [2, 1], 1, False)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_split(x_n_noss_n_axis_n_wr, dtype, tensor_fn, dev, call):
    # smoke test
    x, num_or_size_splits, axis, with_remainder = x_n_noss_n_axis_n_wr
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
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
    true_split = ivy.functional.backends.numpy.split(ivy.to_numpy(x), num_or_size_splits, axis, with_remainder)
    for pred, true in zip(pred_split, true_split):
        assert np.allclose(pred, true)
    # compilation test
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return


# repeat
@pytest.mark.parametrize(
    "x_n_reps_n_axis", [(1, [1], 0), (1, 2, -1), (1, [2], None), ([[0., 1., 2., 3.]], (2, 1, 0, 3), -1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_repeat(x_n_reps_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    x, reps_raw, axis = x_n_reps_n_axis
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    if not isinstance(reps_raw, int) and call is helpers.mx_call:
        # mxnet repeat only supports integer repeats
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    x_shape = list(x.shape)
    if call not in [helpers.jnp_call, helpers.torch_call]:
        # jax and pytorch repeat do not support repeats specified as lists
        ret_from_list = ivy.repeat(x, reps_raw, axis)
    reps = ivy.array(reps_raw, 'int32', dev)
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
                           np.asarray(ivy.functional.backends.numpy.repeat(ivy.to_numpy(x), ivy.to_numpy(reps), axis)))
    else:
        assert np.allclose(call(ivy.repeat, x, reps, axis),
                           np.asarray(ivy.functional.backends.numpy.repeat(ivy.to_numpy(x), ivy.to_numpy(reps), axis)))


# tile
# @pytest.mark.parametrize(
#     "x_n_reps", [(1, [1]), (1, 2), (1, [2]), ([[0., 1., 2., 3.]], (2, 1))])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_tile(x_n_reps, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, reps_raw = x_n_reps
#     if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x, dtype, dev)
#     ret_from_list = ivy.tile(x, reps_raw)
#     reps = ivy.array(reps_raw, 'int32', dev)
#     ret = ivy.tile(x, reps)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     if x.shape == ():
#         expected_shape = tuple(reps_raw) if isinstance(reps_raw, list) else (reps_raw,)
#     else:
#         expected_shape = tuple([int(item * rep) for item, rep in zip(x.shape, reps_raw)])
#     assert ret.shape == expected_shape
#     # value test
#     assert np.allclose(call(ivy.tile, x, reps),
#                        np.asarray(ivy.functional.backends.numpy.tile(ivy.to_numpy(x), ivy.to_numpy(reps))))


# zero_pad
@pytest.mark.parametrize(
    "x_n_pw", [(1, [[1, 1]]), (1, [[0, 0]]), ([[0., 1., 2., 3.]], [[0, 1], [1, 2]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zero_pad(x_n_pw, dtype, tensor_fn, dev, call):
    # smoke test
    x, pw_raw = x_n_pw
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret_from_list = ivy.zero_pad(x, pw_raw)
    pw = ivy.array(pw_raw, 'int32', dev)
    ret = ivy.zero_pad(x, pw)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else x.shape
    expected_shape = tuple([int(item + pw_[0] + pw_[1]) for item, pw_ in zip(x_shape, pw_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.zero_pad, x, pw), ivy.functional.backends.numpy.zero_pad(ivy.to_numpy(x), ivy.to_numpy(pw)))


# fourier_encode
@pytest.mark.parametrize(
    "x_n_mf_n_nb_n_gt", [([2.], 4., 4, [[2.0000000e+00, 1.7484555e-07, 9.9805772e-01,-5.2196848e-01,
                                         3.4969111e-07, 1.0000000e+00, -6.2295943e-02, -8.5296476e-01, 1.0000000e+00]]),
                         ([[1., 2.], [3., 4.], [5., 6.]], [2., 4.], 4,
                          [[[1.0000000e+00, -8.7422777e-08, -8.7422777e-08, -8.7422777e-08,
                             -8.7422777e-08, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00,
                             -1.0000000e+00],
                            [2.0000000e+00, 1.7484555e-07, 9.9805772e-01, -5.2196848e-01,
                             -6.0398321e-07, 1.0000000e+00, -6.2295943e-02, -8.5296476e-01,
                             1.0000000e+00]],
                           [[3.0000000e+00, -2.3849761e-08, -2.3849761e-08, -2.3849761e-08,
                             -2.3849761e-08, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00,
                             -1.0000000e+00],
                            [4.0000000e+00, 3.4969111e-07, -1.2434989e-01, 8.9044148e-01,
                             -1.2079664e-06, 1.0000000e+00, -9.9223840e-01, 4.5509776e-01,
                             1.0000000e+00]],
                           [[5.0000000e+00, -6.7553248e-07, -6.7553248e-07, -6.7553248e-07,
                             -6.7553248e-07, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00,
                             -1.0000000e+00],
                            [6.0000000e+00, 4.7699523e-08, -9.8256493e-01, -9.9706185e-01,
                             -3.7192983e-06, 1.0000000e+00, 1.8591987e-01, 7.6601014e-02,
                             1.0000000e+00]]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_fourier_encode(x_n_mf_n_nb_n_gt, dtype, tensor_fn, dev, call):
    # smoke test
    x, max_freq, num_bands, ground_truth = x_n_mf_n_nb_n_gt
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    if isinstance(max_freq, list):
        max_freq = tensor_fn(max_freq, dtype, dev)
    ret = ivy.fourier_encode(x, max_freq, num_bands)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else list(x.shape)
    expected_shape = x_shape + [1 + 2*num_bands]
    assert list(ret.shape) == expected_shape
    # value test
    assert np.allclose(call(ivy.fourier_encode, x, max_freq, num_bands), np.array(ground_truth), atol=1e-5)


# constant_pad
@pytest.mark.parametrize(
    "x_n_pw_n_val", [(1, [[1, 1]], 1.5), (1, [[0, 0]], -2.7), ([[0., 1., 2., 3.]], [[0, 1], [1, 2]], 11.)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_constant_pad(x_n_pw_n_val, dtype, tensor_fn, dev, call):
    # smoke test
    x, pw_raw, val = x_n_pw_n_val
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret_from_list = ivy.constant_pad(x, pw_raw, val)
    pw = ivy.array(pw_raw, 'int32', dev)
    ret = ivy.constant_pad(x, pw, val)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else x.shape
    expected_shape = tuple([int(item + pw_[0] + pw_[1]) for item, pw_ in zip(x_shape, pw_raw)])
    assert ret.shape == expected_shape
    # value test
    assert np.allclose(call(ivy.constant_pad, x, pw, val),
                       np.asarray(ivy.functional.backends.numpy.constant_pad(ivy.to_numpy(x), ivy.to_numpy(pw), val)))


# swapaxes
@pytest.mark.parametrize(
    "x_n_ax0_n_ax1", [([[1.]], 0, 1), ([[0., 1., 2., 3.]], 1, 0), ([[[0., 1., 2.], [3., 4., 5.]]], -2, -1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_swapaxes(x_n_ax0_n_ax1, dtype, tensor_fn, dev, call):
    # smoke test
    x, ax0, ax1 = x_n_ax0_n_ax1
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.swapaxes(x, ax0, ax1)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    expected_shape = list(x.shape)
    expected_shape[ax0], expected_shape[ax1] = expected_shape[ax1], expected_shape[ax0]
    assert ret.shape == tuple(expected_shape)
    # value test
    assert np.allclose(call(ivy.swapaxes, x, ax0, ax1),
                       np.asarray(ivy.functional.backends.numpy.swapaxes(ivy.to_numpy(x), ax0, ax1)))


# transpose
@pytest.mark.parametrize(
    "x_n_axes", [([[1.]], [1, 0]), ([[0., 1., 2., 3.]], [1, 0]), ([[[0., 1., 2.], [3., 4., 5.]]], [0, 2, 1])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_transpose(x_n_axes, dtype, tensor_fn, dev, call):
    # smoke test
    x, axes = x_n_axes
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.transpose(x, axes)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    x_shape = x.shape
    assert ret.shape == tuple([x.shape[idx] for idx in axes])
    # value test
    assert np.allclose(call(ivy.transpose, x, axes), np.asarray(ivy.functional.backends.numpy.transpose(ivy.to_numpy(x), axes)))


# expand_dims
# @pytest.mark.parametrize(
#     "x_n_axis", [(1., 0), (1., -1), ([1.], 0), ([[0., 1., 2., 3.]], -2), ([[[0., 1., 2.], [3., 4., 5.]]], -3)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_expand_dims(x_n_axis, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, axis = x_n_axis
#     if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x, dtype, dev)
#     ret = ivy.expand_dims(x, axis)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     expected_shape = list(x.shape)
#     expected_shape.insert(axis, 1)
#     assert ret.shape == tuple(expected_shape)
#     # value test
#     assert np.allclose(call(ivy.expand_dims, x, axis), np.asarray(ivy.functional.backends.numpy.expand_dims(ivy.to_numpy(x), axis)))


# where
@pytest.mark.parametrize(
    "cond_n_x1_n_x2", [(True, 2., 3.), (0., 2., 3.), ([True], [2.], [3.]), ([[0.]], [[2., 3.]], [[4., 5.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_where(cond_n_x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    cond, x1, x2 = cond_n_x1_n_x2
    if (isinstance(cond, Number) or isinstance(x1, Number) or isinstance(x2, Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    cond = tensor_fn(cond, dtype, dev)
    x1 = tensor_fn(x1, dtype, dev)
    x2 = tensor_fn(x2, dtype, dev)
    ret = ivy.where(cond, x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.where, cond, x1, x2),
                       np.asarray(ivy.functional.backends.numpy.where(ivy.to_numpy(cond), ivy.to_numpy(x1), ivy.to_numpy(x2))))


# indices_where
@pytest.mark.parametrize(
    "x", [[True], [[0., 1.], [2., 3.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_indices_where(x, dtype, tensor_fn, dev, call):
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.indices_where(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert len(ret.shape) == 2
    assert ret.shape[-1] == len(x.shape)
    # value test
    assert np.allclose(call(ivy.indices_where, x), np.asarray(ivy.functional.backends.numpy.indices_where(ivy.to_numpy(x))))


# isnan
@pytest.mark.parametrize(
    "x_n_res", [([True], [False]),
                ([[0., float('nan')], [float('nan'), 3.]],
                 [[False, True], [True, False]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_isnan(x_n_res, dtype, tensor_fn, dev, call):
    x, res = x_n_res
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.isnan(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.isnan, x), res)


# isinf
@pytest.mark.parametrize(
    "x_n_res", [([True], [False]),
                ([[0., float('inf')], [float('nan'), -float('inf')]],
                 [[False, True], [False, True]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_isinf(x_n_res, dtype, tensor_fn, dev, call):
    x, res = x_n_res
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.isinf(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.isinf, x), res)


# isfinite
@pytest.mark.parametrize(
    "x_n_res", [([True], [True]),
                ([[0., float('inf')], [float('nan'), 3.]],
                 [[True, False], [False, True]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_isfinite(x_n_res, dtype, tensor_fn, dev, call):
    x, res = x_n_res
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.isfinite(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.isfinite, x), res)


# reshape
@pytest.mark.parametrize(
    "x_n_shp", [(1., (1, 1)), (1., 1), (1., []), ([[1.]], []), ([[0., 1.], [2., 3.]], (1, 4, 1))])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_reshape(x_n_shp, dtype, tensor_fn, dev, call):
    # smoke test
    x, new_shape = x_n_shp
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.reshape(x, new_shape)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == ((new_shape,) if isinstance(new_shape, int) else tuple(new_shape))
    # value test
    assert np.allclose(call(ivy.reshape, x, new_shape), np.asarray(ivy.functional.backends.numpy.reshape(ivy.to_numpy(x), new_shape)))


# broadcast_to
@pytest.mark.parametrize(
    "x_n_shp", [([1.], (2, 1)), ([[0., 1.], [2., 3.]], (10, 2, 2))])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_broadcast_to(x_n_shp, dtype, tensor_fn, dev, call):
    # smoke test
    x, new_shape = x_n_shp
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.broadcast_to(x, new_shape)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert len(ret.shape) == len(new_shape)
    # value test
    assert np.allclose(call(ivy.broadcast_to, x, new_shape),
                       np.asarray(ivy.functional.backends.numpy.broadcast_to(ivy.to_numpy(x), new_shape)))


# squeeze
# @pytest.mark.parametrize(
#     "x_n_axis", [(1., 0), (1., -1), ([[1.]], None), ([[[0.], [1.]], [[2.], [3.]]], -1)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_squeeze(x_n_axis, dtype, tensor_fn, dev, call):
#     # smoke test
#     x, axis = x_n_axis
#     if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x, dtype, dev)
#     ret = ivy.squeeze(x, axis)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     if axis is None:
#         expected_shape = [item for item in x.shape if item != 1]
#     elif x.shape == ():
#         expected_shape = []
#     else:
#         expected_shape = list(x.shape)
#         expected_shape.pop(axis)
#     assert ret.shape == tuple(expected_shape)
#     # value test
#     assert np.allclose(call(ivy.squeeze, x, axis), np.asarray(ivy.functional.backends.numpy.squeeze(ivy.to_numpy(x), axis)))


# zeros
# @pytest.mark.parametrize(
#     "shape", [(), (1, 2, 3), tuple([1]*10)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_zeros(shape, dtype, tensor_fn, dev, call):
#     # smoke test
#     ret = ivy.zeros(shape, dtype, dev)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == tuple(shape)
#     # value test
#     assert np.allclose(call(ivy.zeros, shape, dtype, dev), np.asarray(ivy.functional.backends.numpy.zeros(shape, dtype)))


# zeros_like
@pytest.mark.parametrize(
    "x", [1, [1], [[1], [2], [3]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zeros_like(x, dtype, tensor_fn, dev, call):
    # smoke test
    if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.zeros_like(x, dtype, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.zeros_like, x, dtype, dev),
                       np.asarray(ivy.functional.backends.numpy.zeros_like(ivy.to_numpy(x), dtype)))


# ones
# @pytest.mark.parametrize(
#     "shape", [(), (1, 2, 3), tuple([1]*10)])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_ones(shape, dtype, tensor_fn, dev, call):
#     # smoke test
#     ret = ivy.ones(shape, dtype, dev)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == tuple(shape)
#     # value test
#     assert np.allclose(call(ivy.ones, shape, dtype, dev), np.asarray(ivy.functional.backends.numpy.ones(shape, dtype)))


# ones_like
# @pytest.mark.parametrize(
#     "x", [1, [1], [[1], [2], [3]]])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_ones_like(x, dtype, tensor_fn, dev, call):
#     # smoke test
#     if isinstance(x, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
#         # mxnet does not support 0-dimensional variables
#         pytest.skip()
#     x = tensor_fn(x, dtype, dev)
#     ret = ivy.ones_like(x, dtype, dev)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == x.shape
#     # value test
#     assert np.allclose(call(ivy.ones_like, x, dtype, dev),
#                        np.asarray(ivy.functional.backends.numpy.ones_like(ivy.to_numpy(x), dtype)))


# full
# @pytest.mark.parametrize(
#     "shape", [(), (1, 2, 3), tuple([1]*10)])
# @pytest.mark.parametrize(
#     "fill_val", [2., -7.])
# @pytest.mark.parametrize(
#     "dtype", ['float32'])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_full(shape, fill_val, dtype, tensor_fn, dev, call):
#     # smoke test
#     ret = ivy.full(shape, fill_val, dtype, dev)
#     # type test
#     assert ivy.is_array(ret)
#     # cardinality test
#     assert ret.shape == tuple(shape)
#     # value test
#     assert np.allclose(call(ivy.full, shape, fill_val, dtype, dev),
#                        np.asarray(ivy.functional.backends.numpy.full(shape, fill_val, dtype)))


# one_hot
@pytest.mark.parametrize(
    "ind_n_depth", [([0], 1), ([0, 1, 2], 3), ([[1, 3], [0, 0], [8, 4], [7, 9]], 10)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_one_hot(ind_n_depth, dtype, tensor_fn, dev, call):
    # smoke test
    ind, depth = ind_n_depth
    if isinstance(ind, Number) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ind = ivy.array(ind, 'int32', dev)
    ret = ivy.one_hot(ind, depth, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == ind.shape + (depth,)
    # value test
    assert np.allclose(call(ivy.one_hot, ind, depth, dev),
                       np.asarray(ivy.functional.backends.numpy.one_hot(ivy.to_numpy(ind), depth)))


# cross
@pytest.mark.parametrize(
    "x1_n_x2", [([0., 1., 2.], [3., 4., 5.]), ([[0., 1., 2.], [2., 1., 0.]], [[3., 4., 5.], [5., 4., 3.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cross(x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2 = x1_n_x2
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = ivy.array(x1, dtype, dev)
    x2 = ivy.array(x2, dtype, dev)
    ret = ivy.cross(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape
    # value test
    assert np.allclose(call(ivy.cross, x1, x2), np.asarray(ivy.functional.backends.numpy.cross(ivy.to_numpy(x1), ivy.to_numpy(x2))))


# matmul
@pytest.mark.parametrize(
    "x1_n_x2", [([[0., 1., 2.]], [[3.], [4.], [5.]]), ([[0., 1., 2.], [2., 1., 0.]], [[3., 4.], [5., 5.], [4., 3.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_matmul(x1_n_x2, dtype, tensor_fn, dev, call):
    # smoke test
    x1, x2 = x1_n_x2
    if (isinstance(x1, Number) or isinstance(x2, Number)) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x1 = ivy.array(x1, dtype, dev)
    x2 = ivy.array(x2, dtype, dev)
    ret = ivy.matmul(x1, x2)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x1.shape[:-1] + (x2.shape[-1],)
    # value test
    assert np.allclose(call(ivy.matmul, x1, x2), np.asarray(ivy.functional.backends.numpy.matmul(ivy.to_numpy(x1), ivy.to_numpy(x2))))


# cumsum
@pytest.mark.parametrize(
    "x_n_axis", [([[0., 1., 2.]], -1), ([[0., 1., 2.], [2., 1., 0.]], 0), ([[0., 1., 2.], [2., 1., 0.]], 1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cumsum(x_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype, dev)
    ret = ivy.cumsum(x, axis)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cumsum, x, axis), np.asarray(ivy.functional.backends.numpy.cumsum(ivy.to_numpy(x), axis)))


# cumprod
@pytest.mark.parametrize(
    "x_n_axis", [([[0., 1., 2.]], -1), ([[0., 1., 2.], [2., 1., 0.]], 0), ([[0., 1., 2.], [2., 1., 0.]], 1)])
@pytest.mark.parametrize(
    "exclusive", [True, False])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cumprod(x_n_axis, exclusive, dtype, tensor_fn, dev, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype, dev)
    ret = ivy.cumprod(x, axis, exclusive)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.cumprod, x, axis, exclusive),
                       np.asarray(ivy.functional.backends.numpy.cumprod(ivy.to_numpy(x), axis, exclusive)))


# identity
@pytest.mark.parametrize(
    "dim_n_bs", [(3, None), (1, (2, 3)), (5, (1, 2, 3))])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_identity(dim_n_bs, dtype, tensor_fn, dev, call):
    # smoke test
    dim, bs = dim_n_bs
    ret = ivy.identity(dim, dtype, bs, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == (tuple(bs) if bs else ()) + (dim, dim)
    # value test
    assert np.allclose(call(ivy.identity, dim, dtype, bs, dev),
                       np.asarray(ivy.functional.backends.numpy.identity(dim, dtype, bs)))


# meshgrid
@pytest.mark.parametrize(
    "xs", [([1, 2, 3], [4, 5, 6]), ([1, 2, 3], [4, 5, 6, 7], [8, 9])])
@pytest.mark.parametrize(
    "indexing", ['xy', 'ij'])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_meshgrid(xs, indexing, dtype, tensor_fn, dev, call):
    # smoke test
    xs_as_arrays = [ivy.array(x, 'int32', dev) for x in xs]
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
        [np.asarray(i) for i in ivy.functional.backends.numpy.meshgrid(*[ivy.to_numpy(x) for x in xs_as_arrays], indexing=indexing)])


# scatter_flat
@pytest.mark.parametrize(
    "inds_n_upd_n_size_n_tnsr_n_wdup", [([0, 4, 1, 2], [1, 2, 3, 4], 8, None, False),
                                        ([0, 4, 1, 2, 0], [1, 2, 3, 4, 5], 8, None, True),
                                        ([0, 4, 1, 2, 0], [1, 2, 3, 4, 5], None, [11, 10, 9, 8, 7, 6], True)])
@pytest.mark.parametrize(
    "red", ['sum', 'min', 'max', 'replace'])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_flat(inds_n_upd_n_size_n_tnsr_n_wdup, red, dtype, tensor_fn, dev, call):
    # smoke test
    if red in ('sum', 'min', 'max') and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    inds, upd, size, tensor, with_duplicates = inds_n_upd_n_size_n_tnsr_n_wdup
    if ivy.exists(tensor) and call is helpers.mx_call:
        # mxnet does not support scattering into pre-existing tensors
        pytest.skip()
    inds = ivy.array(inds, 'int32', dev)
    upd = tensor_fn(upd, dtype, dev)
    if tensor:
        # pytorch variables do not support in-place updates
        tensor = ivy.array(tensor, dtype, dev) if ivy.current_framework_str() == 'torch'\
            else tensor_fn(tensor, dtype, dev)
    ret = ivy.scatter_flat(inds, upd, size, tensor, red, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if size:
        assert ret.shape == (size,)
    else:
        assert ret.shape == tensor.shape
    # value test
    if red == 'replace' and with_duplicates:
        # replace with duplicates give non-deterministic outputs
        return
    assert np.allclose(call(ivy.scatter_flat, inds, upd, size, tensor, red, dev),
                       np.asarray(ivy.functional.backends.numpy.scatter_flat(
                           ivy.to_numpy(inds), ivy.to_numpy(upd), size,
                           ivy.to_numpy(tensor) if ivy.exists(tensor) else tensor, red)))


# scatter_nd
@pytest.mark.parametrize(
    "inds_n_upd_n_shape_tnsr_n_wdup",
    [([[4], [3], [1], [7]], [9, 10, 11, 12], [8], None, False), ([[0, 1, 2]], [1], [3, 3, 3], None, False),
     ([[0], [2]], [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                   [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], [4, 4, 4], None, False),
     ([[0, 1, 2]], [1], None, [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                               [[4, 5, 6], [7, 8, 9], [1, 2, 3]],
                               [[7, 8, 9], [1, 2, 3], [4, 5, 6]]], False)])
@pytest.mark.parametrize(
    "red", ['sum', 'min', 'max', 'replace'])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_nd(inds_n_upd_n_shape_tnsr_n_wdup, red, dtype, tensor_fn, dev, call):
    # smoke test
    if red in ('sum', 'min', 'max') and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    inds, upd, shape, tensor, with_duplicates = inds_n_upd_n_shape_tnsr_n_wdup
    if ivy.exists(tensor) and call is helpers.mx_call:
        # mxnet does not support scattering into pre-existing tensors
        pytest.skip()
    inds = ivy.array(inds, 'int32', dev)
    upd = tensor_fn(upd, dtype, dev)
    if tensor:
        # pytorch variables do not support in-place updates
        tensor = ivy.array(tensor, dtype, dev) if ivy.current_framework_str() == 'torch'\
            else tensor_fn(tensor, dtype, dev)
    ret = ivy.scatter_nd(inds, upd, shape, tensor, red, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if shape:
        assert tuple(ret.shape) == tuple(shape)
    else:
        assert tuple(ret.shape) == tuple(tensor.shape)
    # value test
    if red == 'replace' and with_duplicates:
        # replace with duplicates give non-deterministic outputs
        return
    ret = call(ivy.scatter_nd, inds, upd, shape, tensor, red, dev)
    true = np.asarray(ivy.functional.backends.numpy.scatter_nd(
                               ivy.to_numpy(inds), ivy.to_numpy(upd), shape,
                               ivy.to_numpy(tensor) if ivy.exists(tensor) else tensor, red))
    assert np.allclose(ret, true)


# gather
@pytest.mark.parametrize(
    "prms_n_inds_n_axis", [([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 4, 7], 0),
                           ([[1, 2], [3, 4]], [[0, 0], [1, 0]], 1)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gather(prms_n_inds_n_axis, dtype, tensor_fn, dev, call):
    # smoke test
    prms, inds, axis = prms_n_inds_n_axis
    prms = tensor_fn(prms, dtype, dev)
    inds = ivy.array(inds, 'int32', dev)
    ret = ivy.gather(prms, inds, axis, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == inds.shape
    # value test
    assert np.allclose(call(ivy.gather, prms, inds, axis, dev),
                       np.asarray(ivy.functional.backends.numpy.gather(ivy.to_numpy(prms), ivy.to_numpy(inds), axis)))


# gather_nd
@pytest.mark.parametrize(
    "prms_n_inds", [([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[0, 1], [1, 0]]),
                    ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[[0, 1]], [[1, 0]]]),
                    ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[[0, 1, 0]], [[1, 0, 1]]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gather_nd(prms_n_inds, dtype, tensor_fn, dev, call):
    # smoke test
    prms, inds = prms_n_inds
    prms = tensor_fn(prms, dtype, dev)
    inds = ivy.array(inds, 'int32', dev)
    ret = ivy.gather_nd(prms, inds, dev)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == inds.shape[:-1] + prms.shape[inds.shape[-1]:]
    # value test
    assert np.allclose(call(ivy.gather_nd, prms, inds, dev),
                       np.asarray(ivy.functional.backends.numpy.gather_nd(ivy.to_numpy(prms), ivy.to_numpy(inds))))


# linear_resample
@pytest.mark.parametrize(
    "x_n_samples_n_axis_n_y_true", [([[10., 9., 8.]], 9, -1, [[10., 9.75, 9.5, 9.25, 9., 8.75, 8.5, 8.25, 8.]]),
                                    ([[[10., 9.], [8., 7.]]], 5, -2,
                                     [[[10., 9.], [9.5, 8.5], [9., 8.], [8.5, 7.5], [8., 7.]]]),
                                    ([[[10., 9.], [8., 7.]]], 5, -1,
                                     [[[10., 9.75, 9.5, 9.25, 9.], [8., 7.75, 7.5, 7.25, 7.]]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_linear_resample(x_n_samples_n_axis_n_y_true, dtype, tensor_fn, dev, call):
    # smoke test
    x, samples, axis, y_true = x_n_samples_n_axis_n_y_true
    x = tensor_fn(x, dtype, dev)
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


# exists
@pytest.mark.parametrize(
    "x", [[1.], None, [[10., 9., 8.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_exists(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev) if x is not None else None
    ret = ivy.exists(x)
    # type test
    assert isinstance(ret, bool)
    # value test
    y_true = x is not None
    assert ret == y_true


# default
@pytest.mark.parametrize(
    "x_n_dv", [([1.], [2.]), (None, [2.]), ([[10., 9., 8.]], [2.])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_default(x_n_dv, dtype, tensor_fn, dev, call):
    x, dv = x_n_dv
    # smoke test
    x = tensor_fn(x, dtype, dev) if x is not None else None
    dv = tensor_fn(dv, dtype, dev)
    ret = ivy.default(x, dv)
    # type test
    assert ivy.is_array(ret)
    # value test
    y_true = ivy.to_numpy(x if x is not None else dv)
    assert np.allclose(call(ivy.default, x, dv), y_true)


# dtype bits
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype_bits(x, dtype, tensor_fn, dev, call):
    # smoke test
    if ivy.invalid_dtype(dtype):
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    ret = ivy.dtype_bits(ivy.dtype(x))
    # type test
    assert isinstance(ret, int)
    assert ret in [1, 8, 16, 32, 64]


# dtype_to_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype", ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype_to_str(x, dtype, tensor_fn, dev, call):
    # smoke test
    if call is helpers.mx_call and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype in ['int64', 'float64']:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    dtype_as_str = ivy.dtype(x, as_str=True)
    dtype_to_str = ivy.dtype_to_str(ivy.dtype(x))
    # type test
    assert isinstance(dtype_as_str, str)
    assert isinstance(dtype_to_str, str)
    # value test
    assert dtype_to_str == dtype_as_str


# dtype_from_str
@pytest.mark.parametrize(
    "x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype", ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_dtype_from_str(x, dtype, tensor_fn, dev, call):
    # smoke test
    if call is helpers.mx_call and dtype == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype in ['int64', 'float64']:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, dev)
    dt0 = ivy.dtype_from_str(ivy.dtype(x, as_str=True))
    dt1 = ivy.dtype(x)
    # value test
    assert dt0 is dt1


def test_cache_fn(dev, call):

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


def test_cache_fn_with_args(dev, call):

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


# def test_framework_setting_with_threading(dev, call):
#
#     if call is helpers.np_call:
#         # Numpy is the conflicting framework being tested against
#         pytest.skip()
#
#     def thread_fn():
#         ivy.set_framework('numpy')
#         x_ = np.array([0., 1., 2.])
#         for _ in range(2000):
#             try:
#                 ivy.reduce_mean(x_)
#             except TypeError:
#                 return False
#         ivy.unset_framework()
#         return True
#
#     # get original framework string and array
#     fws = ivy.current_framework_str()
#     x = ivy.array([0., 1., 2.])
#
#     # start numpy loop thread
#     thread = threading.Thread(target=thread_fn)
#     thread.start()
#
#     # start local original framework loop
#     ivy.set_framework(fws)
#     for _ in range(2000):
#         ivy.reduce_mean(x)
#     ivy.unset_framework()
#
#     assert not thread.join()


def test_framework_setting_with_multiprocessing(dev, call):

    if call is helpers.np_call:
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def worker_fn(out_queue):
        ivy.set_framework('numpy')
        x_ = np.array([0., 1., 2.])
        for _ in range(1000):
            try:
                ivy.mean(x_)
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
        ivy.mean(x)
    ivy.unset_framework()

    worker.join()
    assert output_queue.get_nowait()


# def test_explicit_ivy_framework_handles(dev, call):
#
#     if call is helpers.np_call:
#         # Numpy is the conflicting framework being tested against
#         pytest.skip()
#
#     # store original framework string and unset
#     fw_str = ivy.current_framework_str()
#     ivy.unset_framework()
#
#     # set with explicit handle caught
#     ivy_exp = ivy.get_framework(fw_str)
#     assert ivy_exp.current_framework_str() == fw_str
#
#     # assert backend implemented function is accessible
#     assert 'array' in ivy_exp.__dict__
#     assert callable(ivy_exp.array)
#
#     # assert joint implemented function is also accessible
#     assert 'cache_fn' in ivy_exp.__dict__
#     assert callable(ivy_exp.cache_fn)
#
#     # set global ivy to numpy
#     ivy.set_framework('numpy')
#
#     # assert the explicit handle is still unchanged
#     assert ivy.current_framework_str() == 'numpy'
#     assert ivy_exp.current_framework_str() == fw_str
#
#     # unset global ivy from numpy
#     ivy.unset_framework()


# def test_class_ivy_handles(dev, call):
#
#     if call is helpers.np_call:
#         # Numpy is the conflicting framework being tested against
#         pytest.skip()
#
#     class ArrayGen:
#
#         def __init__(self, ivyh):
#             self._ivy = ivyh
#
#         def get_array(self):
#             return self._ivy.array([0., 1., 2.])
#
#     # create instance
#     ag = ArrayGen(ivy.get_framework())
#
#     # create array from array generator
#     x = ag.get_array()
#
#     # verify this is not a numpy array
#     assert not isinstance(x, np.ndarray)
#
#     # change global framework to numpy
#     ivy.set_framework('numpy')
#
#     # create another array from array generator
#     x = ag.get_array()
#
#     # verify this is not still a numpy array
#     assert not isinstance(x, np.ndarray)


# einops_rearrange
@pytest.mark.parametrize(
    "x_n_pattern_n_newx", [([[0., 1., 2., 3.]], 'b n -> n b', [[0.], [1.], [2.], [3.]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_rearrange(x_n_pattern_n_newx, dtype, tensor_fn, dev, call):
    # smoke test
    x, pattern, new_x = x_n_pattern_n_newx
    x = tensor_fn(x, dtype, dev)
    ret = ivy.einops_rearrange(x, pattern)
    true_ret = einops.rearrange(ivy.to_native(x), pattern)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# einops_reduce
@pytest.mark.parametrize(
    "x_n_pattern_n_red_n_newx", [([[0., 1., 2., 3.]], 'b n -> b', 'mean', [1.5])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_reduce(x_n_pattern_n_red_n_newx, dtype, tensor_fn, dev, call):
    # smoke test
    x, pattern, reduction, new_x = x_n_pattern_n_red_n_newx
    x = tensor_fn(x, dtype, dev)
    ret = ivy.einops_reduce(x, pattern, reduction)
    true_ret = einops.reduce(ivy.to_native(x), pattern, reduction)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# einops_repeat
@pytest.mark.parametrize(
    "x_n_pattern_n_al_n_newx", [([[0., 1., 2., 3.]], 'b n -> b n c', {'c': 2},
                                 [[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_repeat(x_n_pattern_n_al_n_newx, dtype, tensor_fn, dev, call):
    # smoke test
    x, pattern, axes_lengths, new_x = x_n_pattern_n_al_n_newx
    x = tensor_fn(x, dtype, dev)
    ret = ivy.einops_repeat(x, pattern, **axes_lengths)
    true_ret = einops.repeat(ivy.to_native(x), pattern, **axes_lengths)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# profiler
# def test_profiler(dev, call):
#
#     # ToDo: find way to prevent this test from hanging when run alongside other tests in parallel
#
#     # log dir
#     this_dir = os.path.dirname(os.path.realpath(__file__))
#     log_dir = os.path.join(this_dir, '../log')
#
#     # with statement
#     with ivy.Profiler(log_dir):
#         a = ivy.ones([10])
#         b = ivy.zeros([10])
#         a + b
#     if call is helpers.mx_call:
#         time.sleep(1)  # required by MXNet for some reason
#
#     # start and stop methods
#     profiler = ivy.Profiler(log_dir)
#     profiler.start()
#     a = ivy.ones([10])
#     b = ivy.zeros([10])
#     a + b
#     profiler.stop()
#     if call is helpers.mx_call:
#         time.sleep(1)  # required by MXNet for some reason


# container types
def test_container_types(dev, call):
    cont_types = ivy.container_types()
    assert isinstance(cont_types, list)
    for cont_type in cont_types:
        assert hasattr(cont_type, 'keys')
        assert hasattr(cont_type, 'values')
        assert hasattr(cont_type, 'items')


def test_inplace_arrays_supported(dev, call):
    cur_fw = ivy.current_framework_str()
    if cur_fw in ['numpy', 'mxnet', 'torch']:
        assert ivy.inplace_arrays_supported()
    elif cur_fw in ['jax', 'tensorflow']:
        assert not ivy.inplace_arrays_supported()
    else:
        raise Exception('Unrecognized framework')


def test_inplace_variables_supported(dev, call):
    cur_fw = ivy.current_framework_str()
    if cur_fw in ['numpy', 'mxnet', 'torch', 'tensorflow']:
        assert ivy.inplace_variables_supported()
    elif cur_fw in ['jax']:
        assert not ivy.inplace_variables_supported()
    else:
        raise Exception('Unrecognized framework')


# @pytest.mark.parametrize(
#     "x_n_new", [([0., 1., 2.], [2., 1., 0.]), (0., 1.)])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_inplace_update(x_n_new, tensor_fn, dev, call):
#     x_orig, new_val = x_n_new
#     if call is helpers.mx_call and isinstance(x_orig, Number):
#         # MxNet supports neither 0-dim variables nor 0-dim inplace updates
#         pytest.skip()
#     x_orig = tensor_fn(x_orig, 'float32', dev)
#     new_val = tensor_fn(new_val, 'float32', dev)
#     if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or\
#             (tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()):
#         x = ivy.inplace_update(x_orig, new_val)
#         assert id(x) == id(x_orig)
#         assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(new_val))
#         return
#     pytest.skip()


# @pytest.mark.parametrize(
#     "x_n_dec", [([0., 1., 2.], [2., 1., 0.]), (0., 1.)])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_inplace_decrement(x_n_dec, tensor_fn, dev, call):
#     x_orig, dec = x_n_dec
#     if call is helpers.mx_call and isinstance(x_orig, Number):
#         # MxNet supports neither 0-dim variables nor 0-dim inplace updates
#         pytest.skip()
#     x_orig = tensor_fn(x_orig, 'float32', dev)
#     dec = tensor_fn(dec, 'float32', dev)
#     new_val = x_orig - dec
#     if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or\
#             (tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()):
#         x = ivy.inplace_decrement(x_orig, dec)
#         assert id(x) == id(x_orig)
#         assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
#         return
#     pytest.skip()


# @pytest.mark.parametrize(
#     "x_n_inc", [([0., 1., 2.], [2., 1., 0.]), (0., 1.)])
# @pytest.mark.parametrize(
#     "tensor_fn", [ivy.array, helpers.var_fn])
# def test_inplace_increment(x_n_inc, tensor_fn, dev, call):
#     x_orig, inc = x_n_inc
#     if call is helpers.mx_call and isinstance(x_orig, Number):
#         # MxNet supports neither 0-dim variables nor 0-dim inplace updates
#         pytest.skip()
#     x_orig = tensor_fn(x_orig, 'float32', dev)
#     inc = tensor_fn(inc, 'float32', dev)
#     new_val = x_orig + inc
#     if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or\
#             (tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()):
#         x = ivy.inplace_increment(x_orig, inc)
#         assert id(x) == id(x_orig)
#         assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
#         return
#     pytest.skip()
