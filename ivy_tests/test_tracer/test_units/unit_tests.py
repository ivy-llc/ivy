import pytest
import tensorflow as tf
import jax.numpy as jnp
import torch
import numpy as np
import paddle
import gc
from collections import OrderedDict

import ivy
import ivy.tracer.globals as glob
from ivy.tracer import trace_graph
from ivy.tracer.conversion import (
    array_to_new_backend,
    _to_ivy_dtype,
    to_native,
    track,
    _get_ivy_device,
)
from ivy.tracer.source_gen import should_remove_constant


tf_dtypes = [
    tf.uint8,  # # tf.uint16, tf.uint32, tf.uint64,
    # tf.int8, tf.int32, tf.int64, tf.int16,
    tf.float16,  # tf.float32, tf.float64, # tf.bfloat16,
    tf.complex64,  # tf.complex128,
]

torch_dtypes = [
    torch.uint8,
    torch.bool,
    # torch.int8, torch.int16, torch.int32, torch.int64,
    # torch.float16, torch.float32, torch.float64, # torch.bfloat16,
    torch.complex64,  # torch.complex128,
]

np_dtypes = [
    float,  # "i", "f", "f4",
    np.byte,  # np.int8, np.short, np.int16, np.intc, np.int32, np.int_, np.intp, np.int64, # np.longlong,
    np.uint8,  # np.ubyte, # np.ushort, np.uint16, np.uintc, np.uint32, np.uint, np.uintp, np.uint64, # np.ulonglong,
    # np.half, np.float16, np.single, np.float32, np.double, np.float_, np.float64, # np.longdouble,
    # np.csingle, np.cdouble, # np.clongdouble,
    # # np.bool_, np.bool8,
]

jax_dtypes = [
    np.int8,  # float, "i", "f", "f4", # float depends on x64 settings
    # jnp.int8, jnp.int16, jnp.int32, jnp.int_, jnp.int64,
    jnp.uint8,  # jnp.uint16, jnp.uint32, jnp.uint, jnp.uint64,
    # jnp.float16, jnp.single, jnp.float32, jnp.double, jnp.float_, jnp.float64,
    jnp.csingle,  # jnp.cdouble,
    # # jnp.bool_,
]


def _get_wrapped_fns():
    gc.collect()
    ret = {}
    all_objects = gc.get_objects()
    for obj in all_objects:
        if not callable(obj):
            continue
        if "wrapped_for_tracing" in dir(obj):
            ret[obj] = True
    return ret


def check_fns_unwrapped(test):
    def new_test(*args, **kwargs):
        gc.freeze()
        test(*args, **kwargs)
        assert len(_get_wrapped_fns()) <= 1
        gc.unfreeze()

    return new_test


@pytest.mark.parametrize(
    "new_backend", ["numpy", "jax", "paddle", "tensorflow", "torch"]
)
def test_array_to_new_backend(new_backend):
    x = ivy.native_array([1.0, 2.0])
    ivy.set_backend(new_backend)
    new = array_to_new_backend(x, native=True)
    assert ivy.is_native_array(new)
    assert ivy.to_list(new) == [1.0, 2.0]


@pytest.mark.parametrize("dtype", tf_dtypes)
def test_tf_to_ivy_dtype(dtype):
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    ivy_dtype = _to_ivy_dtype(dtype)
    assert isinstance(ivy_dtype, ivy.Dtype)


@pytest.mark.parametrize("dtype", torch_dtypes)
def test_torch_to_ivy_dtype(dtype):
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    ivy_dtype = _to_ivy_dtype(dtype)
    assert isinstance(ivy_dtype, ivy.Dtype)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_np_to_ivy_dtype(dtype):
    if ivy.current_backend_str() != "numpy":
        pytest.skip()
    ivy_dtype = _to_ivy_dtype(dtype)
    assert isinstance(ivy_dtype, ivy.Dtype)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_jax_to_ivy_dtype(dtype):
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    ivy_dtype = _to_ivy_dtype(dtype)
    assert isinstance(ivy_dtype, ivy.Dtype)


@check_fns_unwrapped
def test_unwrapping():
    x = ivy.native_array([1.0])
    trace_graph(lambda x: x + x, args=(x,))


@check_fns_unwrapped
def test_unwrapping_ivy():
    x = ivy.array([1.0])
    trace_graph(lambda x: x + x, args=(x,), to="ivy")


def _fn_which_will_fail(x):
    return ivy.add(x)


@check_fns_unwrapped
def test_unwrapping_on_logging_failure():
    x = ivy.native_array([1.0])
    try:
        trace_graph(_fn_which_will_fail, args=(x,))
    except:
        pass


@check_fns_unwrapped
def test_unwrapping_ivy_on_logging_failure():
    x = ivy.array([1.0])
    try:
        trace_graph(_fn_which_will_fail, args=(x,), to="ivy")
    except:
        pass


def test_not_wrapping_type_hints():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = ivy.array([1.0])
    trace_graph(lambda x: x + x, args=(x,))
    from typing import List
    from ivy.tracer.wrapping import FUNC_TO_PATH

    assert List not in FUNC_TO_PATH


def test_to_native():
    ls = [ivy.array([1.0]), ivy.array([2.0])]
    native_ls = to_native(ls)
    assert id(native_ls) != id(ls)


def test_tracking_dict():
    empty_container = ivy.Container(OrderedDict([]))
    track(empty_container)


# def test_transpiler_overhead_checker():
#     if ivy.current_backend_str() != "torch":
#         pytest.skip()

#     glob.check_transpiler_overhead = True
#     x = torch.tensor([[1.0, 2.0]])
#     graph = graph_transpile(lambda x: x + x, source="torch", to="jax", args=(x,))
#     assert graph.node_expansion
#     assert len(glob.times) == len(glob.transpiled_times)
#     assert len(glob.times) > 0
#     # original time, total transpiled time, n x slower, breakdown of transpiled time
#     for v in glob.times.values():
#         assert len(v) == 4


def test_should_remove_constant():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    assert should_remove_constant(3)
    assert should_remove_constant([(1, None), slice(1, 2, None), Ellipsis, [2.5]])
    assert should_remove_constant([1, 2])
    assert not should_remove_constant(torch.tensor([1.0]))
    assert not should_remove_constant(((None,), (1, [float("inf")])))
    assert not should_remove_constant(-float("inf"))
    assert not should_remove_constant(float("nan"))
    assert not should_remove_constant(slice(tf.constant(1), 2))


@pytest.mark.parametrize(
    "native_array", [tf.constant, jnp.array, torch.tensor, np.array, paddle.to_tensor]
)
def test_get_ivy_device(native_array):
    x = native_array(0)
    assert _get_ivy_device(x) == "cpu"
