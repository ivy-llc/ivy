# # global
# import numpy as np
# from hypothesis import given, assume, strategies as st
#
# # local
# import ivy
# import ivy.functional.backends.numpy as ivy_np
# import ivy_tests.test_ivy.helpers as helpers
#
#
# # cross_entropy
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs, 2),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_cross_entropy(
#     dtype_and_x,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     assume(not any(d in ivy.invalid_dtype_strs for d in dtype))
#     if fw == "torch" and dtype == "float16":
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "cross_entropy",
#         true=np.asarray(x[0], dtype=dtype[0]),
#         pred=np.asarray(x[1], dtype=dtype[1]),
#     )
#
#
# # binary_cross_entropy
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs, 2),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_binary_cross_entropy(
#     dtype_and_x,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     assume(not any(d in ivy.invalid_dtype_strs for d in dtype))
#     if fw == "torch" and dtype == "float16":
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "binary_cross_entropy",
#         true=np.asarray(x[0], dtype=dtype[0]),
#         pred=np.asarray(x[1], dtype=dtype[1]),
#     )
#
#
# # sparse_cross_entropy
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs, 2),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_sparse_cross_entropy(
#     dtype_and_x,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     assume(not any(d in ivy.invalid_dtype_strs for d in dtype))
#     if fw == "torch" and dtype == "float16":
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "sparse_cross_entropy",
#         true=np.asarray(x[0], dtype=dtype[0]),
#         pred=np.asarray(x[1], dtype=dtype[1]),
#     )


# Smoke Tests

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cross_entropy(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, device)
    true = tensor_fn(true, dtype, device)
    ret = ivy.cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(call(ivy.cross_entropy, true, pred), np.asarray(true_target))


# binary_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.7, 0.5]], [[0.35667494, 0.35667494, 0.69314718]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_binary_cross_entropy(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, device)
    true = tensor_fn(true, dtype, device)
    ret = ivy.binary_cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == pred.shape
    # value test
    assert np.allclose(call(ivy.binary_cross_entropy, true, pred), np.asarray(true_target))


# sparse_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([1], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sparse_cross_entropy(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, device)
    true = ivy.array(true, 'int32', device)
    ret = ivy.sparse_cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(call(ivy.sparse_cross_entropy, true, pred), np.asarray(true_target))
