# global
from collections import namedtuple, UserDict
import dill
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import paddle
import pickle
import platform
import pytest
import re
import tensorflow as tf
import time
import torch
from types import ModuleType

# local
import ivy
from ivy.functional.ivy.gradients import _variable
import simple_math_in_ivy as simple_math_in_ivy
import ivy.tracer as tracer
from ivy.tracer.exchange import _convert_dict_to_graph, _convert_graph_to_dict
import ivy.tracer.globals as glob
from ivy.tracer.graph import LazyGraph
from ivy.tracer.numpy_proxy import NewNDArray
from ivy.tracer import trace_graph

glob.use_reloader = False

IS_MAC_ARM = platform.system() == "Darwin" and platform.machine() == "arm64"
IS_WINDOWS = platform.system() == "Windows"


def _inplace_fn(x):
    for _ in range(100000):
        pass
    return (x + 10) ** 0.5 - 5


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_trace_inplace(
    x_raw,
    dtype,
    array_caching,
    native_array,
    dev,
):
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    x = ivy.to_native(x) if native_array else x

    fname = "fn2_inplace_{}".format(array_caching)
    graph = trace_graph(
        _inplace_fn,
        array_caching=array_caching,
        args=(x,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    _inplace_fn(x)  # warmup for jax's jit
    start_time = time.perf_counter()
    non_traced_return = ivy.to_numpy(_inplace_fn(x))
    non_comp_time_taken = time.perf_counter() - start_time
    graph(x)  # warmup for jax's jit
    start_time = time.perf_counter()
    traced_return = ivy.to_numpy(graph(x))
    comp_time_taken = time.perf_counter() - start_time
    assert comp_time_taken < non_comp_time_taken
    assert np.allclose(non_traced_return, traced_return)


# functional


def _functional_fn(x):
    x += ivy.array([1.0])
    y = ivy.mean(x, keepdims=True)
    z = ivy.mean(x, keepdims=True)
    f = ivy.mean(y, keepdims=True)
    time.sleep(0.05)
    k = ivy.cos(z)
    m = ivy.sin(f)
    o = ivy.tan(y)
    return ivy.concat([k, m, o], axis=-1)


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_graph(
    x_raw,
    dtype,
    array_caching,
    dev,
):
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)

    fname = "fn4_{}".format(array_caching)
    graph = trace_graph(
        _functional_fn,
        array_caching=array_caching,
        args=(x,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    start_time = time.perf_counter()
    non_traced_return = ivy.to_numpy(_functional_fn(x))
    non_traced_time_taken = time.perf_counter() - start_time
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    start_time = time.perf_counter()
    traced_return = ivy.to_numpy(graph(x))
    traced_time_taken = time.perf_counter() - start_time
    assert traced_time_taken < non_traced_time_taken
    assert np.allclose(non_traced_return, traced_return)


@pytest.mark.parametrize("x_raw", [[1.0]])
def test_trace_to(x_raw, dev):
    x = ivy.array(x_raw, device=dev)
    to = ivy.current_backend_str()
    ivy.unset_backend()
    graph = trace_graph(
        _functional_fn,
        to=to,
        args=(x,),
    )
    graph.show()
    ivy.set_backend(to)
    # value test
    x = ivy.array(x_raw, device=dev)
    non_traced_return = ivy.to_numpy(_functional_fn(x))
    x = ivy.array(x_raw, device=dev)
    traced_return = ivy.to_numpy(graph(x))
    assert np.allclose(non_traced_return, traced_return)


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_torch_native_compilation_w_mode(
    x_raw,
    dtype,
    array_caching,
    mode,
    dev,
):
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)

    native_trace_fn = trace_graph(
        _functional_fn,
        array_caching=array_caching,
        compile_mode=mode,
        args=(x,),
    )

    # value test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    non_traced_return = ivy.to_numpy(_functional_fn(x))
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    traced_return = ivy.to_numpy(native_trace_fn(ivy.to_native(x, nested=True)))
    assert np.allclose(non_traced_return, traced_return)


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_native_compilation(
    x_raw,
    dtype,
    array_caching,
    dev,
):
    pytest.skip()  # skipping due to torch.compile problem on actions
    if ivy.current_backend_str() == "numpy":
        pytest.skip()
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)

    native_trace_fn = trace_graph(
        _functional_fn,
        array_caching=array_caching,
        args=(x,),
        backend_compile=True,
        graph_caching=False,
    )

    # value test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    non_traced_return = ivy.to_numpy(_functional_fn(x))
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    traced_return = ivy.to_numpy(native_trace_fn(ivy.to_native(x, nested=True)))
    assert np.allclose(non_traced_return, traced_return)


@pytest.mark.parametrize("array_caching", [True, False])
@pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch", "paddle"])
def test_trace_to_ivy(array_caching, backend):
    x = ivy.array([1.0, 2.0])
    non_traced_ret = ivy.to_numpy(_functional_fn(x))
    traced_fn = trace_graph(
        _functional_fn,
        to="ivy",
        array_caching=array_caching,
        args=(x,),
    )
    ivy.set_backend(backend)
    x = ivy.array([1.0, 2.0])
    traced_ret = ivy.to_numpy(traced_fn(x))
    assert np.allclose(non_traced_ret, traced_ret)


# random


def _rand_fn(x):
    return x + ivy.random_uniform(low=0.0, high=1.0, shape=x.shape)


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("include_generators", [True, False])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_random(
    x_raw,
    dtype,
    include_generators,
    array_caching,
    dev,
):
    if not include_generators:
        pytest.skip()
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    fname = "w_random_{}_{}".format(include_generators, array_caching)
    graph = trace_graph(
        _rand_fn,
        include_generators=include_generators,
        array_caching=array_caching,
        args=(x,),
    )
    # graph.show(
    #     output_connected_only=False,
    #     # fname=fname + ".html",  # uncomment this to save the graph locally
    # )
    # value test
    nc_return0 = _rand_fn(x)
    nc_return1 = _rand_fn(x)
    assert nc_return0 != nc_return1
    c_return0 = graph(x)
    c_return1 = graph(x)
    if include_generators:
        assert c_return0 != c_return1


@pytest.mark.parametrize("include_generators", [True, False])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_random_to_ivy(include_generators, array_caching, dev):
    if not include_generators:
        pytest.skip()
    # smoke test
    x = ivy.array([1.0], dtype="float32", device=dev)
    fname = "w_random_{}_{}".format(include_generators, array_caching)
    graph = trace_graph(
        _rand_fn,
        to="ivy",
        include_generators=include_generators,
        array_caching=array_caching,
        args=(x,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_return0 = _rand_fn(x)
    nc_return1 = _rand_fn(x)
    assert nc_return0 != nc_return1
    c_return0 = graph(x)
    c_return1 = graph(x)
    if include_generators:
        assert c_return0 != c_return1


# detached divide


def _detach_div_fn(x):
    return x + (ivy.array([1.0]) / ivy.array([2.0]))


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_detached_divide(x_raw, dtype, array_caching, dev):
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    fname = "w_detached_divide_{}".format(array_caching)
    graph = trace_graph(
        _detach_div_fn,
        array_caching=array_caching,
        args=(x,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_return = ivy.to_numpy(_detach_div_fn(x))
    c_return = ivy.to_numpy(graph(x))
    assert np.allclose(nc_return, c_return)


# setitem and getitem


def setitem_getitem(x):
    a = x[2]
    x[1] = a
    return x


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_setitem_getitem(dtype, array_caching, dev):
    # framework specific trace test
    x1 = ivy.array([1, 2, 3], dtype=dtype, device=dev)
    x2 = ivy.array([10, 20, 30], dtype=dtype, device=dev)
    x3 = ivy.array([10, 20, 30], dtype=dtype, device=dev)
    fname = "setitem_getitem_{}".format(array_caching)
    graph = trace_graph(
        setitem_getitem,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_return = ivy.to_numpy(setitem_getitem(x2))
    c_return = ivy.to_numpy(graph(x3))
    assert np.allclose(nc_return, c_return)


def setitem_getitem_jax(x):
    a = x[2]
    x = x.at[1].set(a)
    return x


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_setitem_getitem_backend_specific(dtype, array_caching, dev):
    # framework specific trace test
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    x1 = ivy.native_array([1, 2, 3], dtype=dtype)
    x2 = ivy.native_array([10, 20, 30], dtype=dtype)
    x3 = ivy.native_array([10, 20, 30], dtype=dtype)
    fname = "setitem_getitem_backend_specific_{}".format(array_caching)
    graph = trace_graph(
        setitem_getitem_jax,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_return = setitem_getitem_jax(x2)
    c_return = graph(x3)
    assert np.allclose(nc_return, c_return)


# input in output


def _input_in_output(x, y):
    return x + 2, y


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_input_in_output(x_raw, dtype, array_caching, dev):
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    y = ivy.array(x_raw, dtype=dtype, device=dev)
    fname = "input_in_output_{}".format(array_caching)
    graph = trace_graph(
        _input_in_output,
        array_caching=array_caching,
        args=(x, y),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_ret_a, nc_ret_b = _input_in_output(x, y)
    c_ret_a, c_ret_b = graph(x, y)
    assert np.allclose(ivy.to_numpy(nc_ret_a), ivy.to_numpy(c_ret_a))
    assert np.allclose(ivy.to_numpy(nc_ret_b), ivy.to_numpy(c_ret_b))


def _container_input_in_output(x):
    return x


def test_container_input_in_output():
    x = ivy.Container(a=ivy.native_array([0.0]))
    graph = trace_graph(_container_input_in_output, args=(x,))
    x = ivy.Container(a=ivy.native_array([1.0]))
    non_traced_ret = _container_input_in_output(x)
    traced_ret = graph(x)
    assert np.allclose(ivy.to_numpy(traced_ret.a), ivy.to_numpy(non_traced_ret.a))


# inplace variable update


def _inplace_var_update(weight, grad):
    ivy.inplace_decrement(weight, grad)
    return weight


@pytest.mark.parametrize("weight_n_grad", [([1], [2])])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_inplace_var_update(
    weight_n_grad,
    dtype,
    array_caching,
    dev,
):
    # raw values
    weight_raw, grad_raw = weight_n_grad
    # as tensors
    weight = _variable(ivy.array(weight_raw, dtype=dtype, device=dev))
    # trace
    fname = "inplace_var_update_{}".format(array_caching)
    graph = trace_graph(
        _inplace_var_update,
        array_caching=array_caching,
        args=(weight, ivy.copy_array(weight)),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_weight = _variable(ivy.array(weight_raw, dtype=dtype, device=dev))
    grad = ivy.array(grad_raw, dtype=dtype, device=dev)
    nc_new_weight = _inplace_var_update(nc_weight, grad)
    c_weight = _variable(ivy.array(weight_raw, dtype=dtype, device=dev))
    grad = ivy.array(grad_raw, dtype=dtype, device=dev)
    c_new_weight = graph(c_weight, grad)
    assert not np.allclose(weight_raw, ivy.to_numpy(nc_new_weight))
    assert not np.allclose(weight_raw, ivy.to_numpy(c_new_weight))
    assert np.allclose(ivy.to_numpy(nc_new_weight), ivy.to_numpy(c_new_weight))


# with stateful


class Stateful:
    def __init__(self, with_array_caching, dev):
        self._with_array_caching = with_array_caching
        self._state = (
            ivy.native_array([0.0], device=dev)
            if ivy.current_backend_str() != "numpy"
            else NewNDArray([0.0])
        )

    def forward(self, _x):
        self._state = self._state + 1
        return _x + self._state

    def trace_graph(self, _x):
        fname = "w_stateful_{}".format(self._with_array_caching)
        # noinspection PyAttributeOutsideInit
        self.forward = trace_graph(
            self.forward,
            stateful=[self],
            array_caching=self._with_array_caching,
            args=(_x,),
        )
        self.forward.show(
            output_connected_only=False,
            # fname=fname + ".html",  # uncomment this to save the graph locally
        )

    def __setattr__(self, key, value):
        self.__dict__[key] = value


@pytest.mark.parametrize("x_raw", [([0])])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_stateful(
    x_raw,
    dtype,
    array_caching,
    dev,
):
    # as tensors
    x = ivy.native_array(x_raw, dtype=dtype, device=dev)

    # non-traced
    stateful = Stateful(array_caching, dev)
    nc_ret_0 = stateful.forward(x)
    assert nc_ret_0 == x + 1
    nc_ret_1 = stateful.forward(x)
    assert nc_ret_1 == x + 2
    nc_ret_2 = stateful.forward(x)
    assert nc_ret_2 == x + 3

    # traced
    stateful = Stateful(array_caching, dev)
    stateful.trace_graph(x)
    c_ret_0 = stateful.forward(x)
    assert c_ret_0 == x + 1
    c_ret_1 = stateful.forward(x)
    assert c_ret_1 == x + 2
    c_ret_2 = stateful.forward(x)
    assert c_ret_2 == x + 3


# with stateful container


def _update_container(cont, x):
    cont.new_attribute = 2
    return x + 1


class Untracked:
    pass


@pytest.mark.parametrize("x_raw", [([0])])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_stateful_cont(
    x_raw,
    dtype,
    array_caching,
    dev,
):
    # as tensors
    x = ivy.array(x_raw, dtype=dtype, device=dev)

    # non-traced
    cont = ivy.Container(x=Untracked())
    assert not hasattr(cont, "new_attribute")
    nc_ret_0 = _update_container(cont, x)
    assert nc_ret_0 == x + 1
    assert hasattr(cont, "new_attribute")
    assert cont.new_attribute == 2

    # traced
    cont = ivy.Container(x=Untracked())
    graph = trace_graph(
        _update_container,
        arg_stateful_idxs=[[0]],
        array_caching=array_caching,
        args=(cont.cont_deep_copy(), x),
    )
    assert not hasattr(cont, "new_attribute")
    c_ret_0 = graph(cont, x)
    assert x + 1 == c_ret_0
    assert hasattr(cont, "new_attribute")
    assert cont.new_attribute == 2


# side effects from compiling stateful


class StatefulEffect:
    def __init__(self, cont, dev):
        self.cont = cont
        self.device = dev

    def add_one(self, x):
        self.cont.x = ivy.array([21.0], device=self.device)  # this is a side effect
        return x + 1


def _add_one_w_effect(s, x):
    return s.add_one(x)


@pytest.mark.parametrize("x_raw", [([0])])
@pytest.mark.parametrize("y_raw", [([0])])
@pytest.mark.parametrize("dtype_str", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_stateful_effect(x_raw, y_raw, dtype_str, array_caching, dev):
    # without trace
    x = ivy.array(x_raw, dtype=dtype_str, device=dev)
    y = ivy.array(y_raw, dtype=dtype_str, device=dev)
    cont = ivy.Container(x=y)
    s = StatefulEffect(cont, dev)
    assert cont.x == 0
    assert x == 0
    x = _add_one_w_effect(s, x)
    assert cont.x == 21
    assert x == 1

    # with trace
    x = ivy.array(x_raw, dtype=dtype_str, device=dev)
    y = ivy.array(y_raw, dtype=dtype_str, device=dev)
    cont = ivy.Container(x=y)
    s = StatefulEffect(cont, dev)
    assert cont.x == 0
    assert x == 0
    fn = trace_graph(
        _add_one_w_effect,
        arg_stateful_idxs=[[0]],
        array_caching=array_caching,
        args=(s, x),
    )
    assert cont.x == 0  # cont should not be changed by compilation
    assert x == 0


# with stateful in args


class StatefulInArg:
    def __init__(self, dev):
        self._state = ivy.native_array([0.0], device=dev)

    def add_one(self):
        self._state += 1

    @property
    def state(self):
        return self._state


def _stateful_in_arg_method(x, sia):
    x = x + 1
    sia.add_one()
    return x


@pytest.mark.parametrize("x_raw", [([0])])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_stateful_in_args(x_raw, dtype, array_caching, dev):
    # as tensors
    x = ivy.native_array(x_raw, dtype=dtype, device=dev)
    sia = StatefulInArg(dev)
    # trace
    fname = "w_stateful_in_args_{}".format(array_caching)
    graph = trace_graph(
        _stateful_in_arg_method,
        arg_stateful_idxs=[[1]],
        array_caching=array_caching,
        args=(x, sia),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    ret = graph(x, sia)
    assert ret == 1
    assert sia.state == 1


# trainable modules


def _trainable_module(x, trainable_module):
    return trainable_module(x)


class TrainableTorchModule(torch.nn.Module):
    def __init__(self, in_size=2, out_size=2, intermediate=3):
        super(TrainableTorchModule, self).__init__()
        self.fc1 = torch.nn.Linear(in_size, intermediate)
        self.fc2 = torch.nn.Linear(intermediate, out_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        output = self.tanh(self.fc1(x))
        output = self.tanh(self.fc2(output))
        return output


@pytest.mark.parametrize("to", ["torch"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_w_trainable_torch_module(array_caching, to):
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = torch.tensor([1.0, 2.0])
    ttm = TrainableTorchModule()
    # trace_graph
    graph = trace_graph(
        ttm,
        to=to,
        array_caching=array_caching,
        args=(x,),
    )
    # type test
    assert isinstance(graph, torch.nn.Module)
    # value test
    nc_ret = _trainable_module(x, ttm)
    c_ret = _trainable_module(x, graph)
    assert np.allclose(nc_ret.detach().numpy(), c_ret.detach().numpy())


# reverse built-in


def _rev_builtin(x):
    return 10.0**x


@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_reverse_builtin(array_caching):
    # config
    x = ivy.array([0.0, 1.0, 2.0])
    # trace
    fname = "rev_builtin_{}".format(array_caching)
    graph = trace_graph(
        _rev_builtin,
        array_caching=array_caching,
        args=(x,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_ret = ivy.to_numpy(_rev_builtin(x))
    c_ret = ivy.to_numpy(graph(x))
    assert np.allclose(nc_ret, c_ret)


# tuples in output


def _tuples_out(x):
    a = x + x
    return (a, (a, a))


@pytest.mark.parametrize("array_caching", [True, False])
def test_trace_tuples_out(array_caching):
    # config
    x = ivy.array([0.0, 1.0, 2.0])
    # trace
    fname = "tuples_out_{}".format(array_caching)
    graph = trace_graph(
        _tuples_out,
        array_caching=array_caching,
        args=(x,),
    )
    # value test
    c_ret = graph(x)
    assert isinstance(c_ret[1], tuple)


def _fn_w_failing_try(x):
    try:
        # will always fail
        tf.add(x)
    except:
        return tf.add(x, x)


def test_wrapped_fn_failing_in_try_block():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    x = tf.constant([1.0, 2.0])
    traced_fn = trace_graph(_fn_w_failing_try, args=(x,))
    traced_ret = traced_fn(x)
    non_traced_ret = _fn_w_failing_try(x)
    assert np.allclose(traced_ret, non_traced_ret)


def test_cont_with_NDArray():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()
    x1, x2 = ivy.array([2], dtype="int16"), ivy.array([3], dtype="uint64")
    x1, x2 = ivy.Container(a=x1, b=ivy.Container(c=x1, d=x1)), ivy.Container(
        a=x2, b=ivy.Container(c=x2, d=x2)
    )
    alpha = 1
    args = (x1, x2)
    kwargs = {"alpha": alpha}
    non_traced_ret = ivy.add(*args, **kwargs)
    fn = trace_graph(ivy.add, args=args, kwargs=kwargs)
    traced_ret = fn(*args, **kwargs)
    assert all(traced_ret == non_traced_ret)


def _fn_which_will_fail(x):
    # since add needs 2 args
    return ivy.add(x)


def test_resetting_globals_when_compiling_fails():
    x = ivy.array([1.0, 2.0])
    try:
        trace_graph(_fn_which_will_fail, args=(x,))
    except:
        pass

    assert glob.tracing_paused == True
    assert glob.tracing_stack == list()
    assert glob.transformed_callables == list()
    assert glob.iterator_chains == dict()
    assert glob.raw_id_to_weakref == dict()
    assert glob.raw_id_to_unique_id["train"] == dict()
    assert glob.raw_id_to_unique_id["eval"] == dict()
    assert glob.dependent_ids["train"] == set()
    assert glob.dependent_ids["eval"] == set()


def _lazy_fn(x):
    return ivy.sum(x)


def test_lazy_trace_graph():
    x = ivy.array([1.0, 2.0])
    eager_graph = trace_graph(_lazy_fn, args=(x,))
    assert isinstance(eager_graph, tracer.graph.Graph)
    lazy_graph = trace_graph(_lazy_fn)
    assert isinstance(lazy_graph, tracer.graph.LazyGraph)
    assert lazy_graph._initialized == False
    lazy_graph(x)
    assert lazy_graph._initialized


def test_lazy_trace_decorated():
    x = ivy.array([1.0, 2.0])

    @trace_graph
    def _decorated_lazy_fn(x):
        return ivy.sum(x)

    assert isinstance(_decorated_lazy_fn, tracer.graph.LazyGraph)
    assert _decorated_lazy_fn._initialized == False
    _decorated_lazy_fn(x)
    assert _decorated_lazy_fn._initialized


def test_lazy_trace_multiple():
    x = ivy.array([1.0, 2.0])

    graphs = trace_graph(_lazy_fn, _lazy_fn, _lazy_fn)
    for g in graphs:
        assert isinstance(g, tracer.graph.LazyGraph)
        assert g._initialized == False
        g(x)
        assert g._initialized


def test_lazy_trace_module():
    x = ivy.array([1.0, 2.0])
    y = ivy.array([3.0, 4.0])

    traced_simple_math = trace_graph(simple_math_in_ivy)
    assert isinstance(traced_simple_math, ModuleType)
    assert isinstance(traced_simple_math.add, tracer.graph.LazyGraph)
    assert traced_simple_math.add._initialized == False
    traced_simple_math.add(x, y)
    assert traced_simple_math.add._initialized


def test_trace_library_submodules():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    import kornia

    # trace library lazily
    comp_kornia = trace_graph(kornia)
    # check that submodules are transpiled correctly
    assert isinstance(comp_kornia.filters.canny, LazyGraph)
    assert isinstance(comp_kornia.utils.image.image_to_tensor, LazyGraph)


def _fn_with_out(*args, **kwargs):
    return ivy.add(*args, **kwargs)


def test_out_kwarg():
    x1, x2 = ivy.array([0], dtype="int32"), ivy.array([0], dtype="int32")
    out = ivy.array([0], dtype="int32")
    args = (x1, x2)
    kwargs = {"out": out}
    graph = trace_graph(_fn_with_out, to="ivy", args=args, kwargs=kwargs)
    comp_ret = graph(*args, **kwargs)
    assert comp_ret is out


def _ivy_fn_w_native_dtype(x):
    return ivy.sum(x, dtype=ivy.as_native_dtype(x.dtype))


# when compiling to ivy, we need to make sure any dtype arguments being cached
# are cached as ivy dtypes, so when the backend is changed there are no issues
# with passing torch dtypes to jax functions for example
def test_caching_ivy_dtypes():
    x = ivy.array([1.0, 2.0])
    traced_fn = trace_graph(_ivy_fn_w_native_dtype, to="ivy", args=(x,))
    non_traced_ret = ivy.to_numpy(_ivy_fn_w_native_dtype(x))
    ivy.set_backend("jax")
    assert np.allclose(ivy.to_numpy(traced_fn(x)), non_traced_ret)
    ivy.set_backend("torch")
    assert np.allclose(ivy.to_numpy(traced_fn(x)), non_traced_ret)
    ivy.set_backend("tensorflow")
    assert np.allclose(ivy.to_numpy(traced_fn(x)), non_traced_ret)
    ivy.set_backend("numpy")
    assert np.allclose(ivy.to_numpy(traced_fn(x)), non_traced_ret)
    ivy.set_backend("paddle")
    assert np.allclose(ivy.to_numpy(traced_fn(x)), non_traced_ret)


def _vmap_fn(x):
    return jnp.sum(x)


def _vmap(x):
    vmap_fn = jax.vmap(_vmap_fn)
    z = vmap_fn(x)
    y = jnp.sum(x) + z
    return y


# ToDo: Add transpilation test
# @pytest.mark.parametrize("array_caching", [True])  # FixMe: False
# def test_vmap(array_caching):
#     if ivy.current_backend_str() != "jax":
#         pytest.skip()
#     # config
#     x1 = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     x2 = jnp.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
#     # trace
#     fname = "vmap_{}".format(array_caching)
#     graph = trace_graph(
#         _vmap,
#         array_caching=array_caching,
#         args=(x1,),
#     )
#     graph.show(
#         output_connected_only=False,
#         # fname=fname + ".html",  # uncomment this to save the graph locally
#     )
#     # value test
#     # This works bc the vmap result is being cached (only with array_caching=True)
#     nc_ret = _vmap(x1)
#     c_ret = graph(x1)
#     assert np.allclose(nc_ret, c_ret)
#     # FixMe: This won't work
#     nc_ret = _vmap(x2)
#     c_ret = graph(x2)
#     assert np.allclose(nc_ret, c_ret)


def _test_numpy_scalars(x):
    y = np.abs(x)
    return -y


numpy_scalars = [
    np.float64,
    np.float32,
    np.float16,
    np.complex128,
    np.complex64,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.uint64,
    np.uint32,
    np.uint16,
    np.uint8,
]

if not IS_MAC_ARM and not IS_WINDOWS:
    numpy_scalars.extend([np.float128, np.complex256])


@pytest.mark.parametrize(
    "scalar_type",
    numpy_scalars,
)
def test_numpy_scalar_tracking(scalar_type):
    if ivy.current_backend_str() != "numpy":
        pytest.skip()

    x = scalar_type(1)
    graph = trace_graph(_test_numpy_scalars, args=(x,))
    traced_ret = graph(x)
    non_traced_ret = _test_numpy_scalars(x)
    assert traced_ret == non_traced_ret
    assert type(traced_ret) == type(non_traced_ret)


def _inplace_torch_fn(x):
    y = x[0]
    z = x[1]
    y.add_(1)
    z *= 3
    w = z + z
    x += 1
    return x


def test_fn_w_inplace_torch():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = ivy.native_array([3.0, 4.0])
    non_traced_ret = ivy.to_numpy(_inplace_torch_fn(x))
    x = ivy.native_array([3.0, 4.0])
    graph = trace_graph(_inplace_torch_fn, args=(x,))
    x = ivy.native_array([3.0, 4.0])
    traced_ret = ivy.to_numpy(graph(x))
    assert np.allclose(non_traced_ret, traced_ret)


def _fn_takes_user_dict(input):
    x = input["input"]
    return ivy.add(x, x)


def test_user_dict_input():
    inputs = UserDict({"input": ivy.array([1.0, 2.0, 3.0])})
    graph = trace_graph(_fn_takes_user_dict, args=(inputs,))
    inputs = UserDict({"input": ivy.array([2.0, 3.0, 4.0])})
    traced_ret = ivy.to_numpy(graph(inputs))
    non_traced_ret = ivy.to_numpy(_fn_takes_user_dict(inputs))
    assert np.allclose(traced_ret, non_traced_ret)


@jax.jit
def _jitted_fn(x):
    return jnp.add(x, x)


def _some_jax_fn(x):
    y = _jitted_fn(x)
    return jnp.sin(y)


def test_compiling_with_jax_jit():
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    x = ivy.native_array([1.0, 2.0])
    y = ivy.native_array([2.0, 3.0])
    non_traced_ret = ivy.to_numpy(_some_jax_fn(y))
    graph = trace_graph(_some_jax_fn, args=(x,))
    traced_ret = ivy.to_numpy(graph(y))
    assert np.allclose(traced_ret, non_traced_ret)


def _paddle_numpy_function(x):
    z = paddle.multiply(x, x)
    return (z + z).numpy() ** 2


@pytest.mark.parametrize("array_caching", [True, False])
@pytest.mark.parametrize("include_generators", [True, False])
def test_trace_paddle_with_numpy(array_caching, include_generators):
    if ivy.current_backend_str() != "paddle" or (
        not include_generators and not array_caching
    ):
        pytest.skip()

    x = np.array([1.0, 2.0])
    graph = trace_graph(
        _paddle_numpy_function,
        args=(x,),
        with_numpy=True,
        array_caching=array_caching,
        include_generators=include_generators,
    )
    y = paddle.to_tensor([2.0, 3.0])
    non_traced_ret = _paddle_numpy_function(y)
    traced_ret = graph(y)
    assert np.allclose(non_traced_ret, traced_ret)


def _fn_w_copyto(x, y):
    z = np.add(y, y)
    np.copyto(x, z)
    return x


def test_numpy_copyto():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    graph = trace_graph(_fn_w_copyto, args=(x, y))
    x = np.array([1.0, 2.0])
    traced_ret = graph(x, y)
    x = np.array([1.0, 2.0])
    non_traced_ret = _fn_w_copyto(x, y)
    assert np.allclose(traced_ret, non_traced_ret)


def test_jax_bfloat_unaffected_by_wrapping():
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    x = jnp.array(1.0)
    trace_graph(lambda x: jnp.exp(x), args=(x,))
    print(jnp.array(1.0, dtype="bfloat16"))


def test_trace_compatibility_with_tf_function():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()

    def func(x):
        return x**2 + x

    x = tf.ones((2, 3, 2))
    y = tf.random.normal((1, 2, 2))
    graph = trace_graph(func, args=(x,))

    nc_ret = func(y)
    c_ret = graph(y)
    tf_func_c_ret = tf.function(graph, autograph=False)(y)
    tf_func_c_ret_auto = tf.function(graph)(y)

    assert np.allclose(nc_ret, c_ret)
    assert np.allclose(tf_func_c_ret, tf_func_c_ret_auto)
    assert np.allclose(nc_ret, tf_func_c_ret_auto)
    assert np.allclose(c_ret, tf_func_c_ret)
    assert np.allclose(c_ret, tf_func_c_ret_auto)
    assert np.allclose(nc_ret, tf_func_c_ret)


def _fn_with_list_input(list_):
    data = ivy.concat(list_, axis=-1)
    return ivy.add(list_[0], list_[1])


def test_trace_doesnt_inplace_update_lists():
    x = ivy.random_uniform(shape=(1,))
    graph = trace_graph(_fn_with_list_input, to="ivy", args=([x, x],))

    y = ivy.random_uniform(shape=(1,))
    original_ret = ivy.to_numpy(_fn_with_list_input([y, y]))
    traced_ret = ivy.to_numpy(graph([y, y]))
    assert np.allclose(original_ret, traced_ret)


def _fn_with_inplacing_cached_arg(x):
    a = torch.ones((2,))
    return a.add_(x)


def test_inplace_updating_cached_arg():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.tensor([1.0, 2.0])
    graph = trace_graph(_fn_with_inplacing_cached_arg, args=(x,))
    original_ret = ivy.to_numpy(_fn_with_inplacing_cached_arg(x))
    traced_ret = ivy.to_numpy(graph(x))
    assert np.allclose(original_ret, traced_ret)


def _fn_with_named_tuples(x):
    _min = x.min(dim=0)
    return _min.indices, x.max(dim=0)


def test_return_types_torch_in_output():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.tensor([1.0])
    graph = trace_graph(_fn_with_named_tuples, args=(x,))
    ret = _fn_with_named_tuples(x)
    comp_ret = graph(x)
    assert ret[0] == comp_ret[0]
    assert ret[1] == comp_ret[1]
    if torch.__version__ >= "1.11.0":
        assert isinstance(comp_ret[1], torch.return_types.max)


def _single_subclass_output(x):
    return x.sort()


def test_single_subclass_output():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.tensor([1, 3, 2, 4])
    graph = trace_graph(_single_subclass_output, args=(x,))
    ret = _single_subclass_output(x)
    comp_ret = graph(x)
    assert np.allclose(ret[0], comp_ret[0])
    assert np.allclose(ret[1], comp_ret[1])
    assert isinstance(comp_ret, torch.return_types.sort)


def fn_with_raw_dict_output(key1, value1):
    return {key1: value1}


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
def test_trace_fn_with_raw_dict_output(
    fw,
):
    ivy.set_backend(fw)
    # raw values
    c_key = nc_key = "a"
    c_value, nc_value = 10, 20
    # trace
    fname = "fn_with_raw_dict_output_{}".format(fw)
    ## int
    graph = trace_graph(
        fn_with_raw_dict_output,
        args=(nc_key, nc_value),
        static_argnums=[0],
    )
    # value test
    nc_ret = fn_with_raw_dict_output(c_key, c_value)
    c_ret = graph(c_key, c_value)
    assert nc_ret == c_ret


def _fn_with_arg_to_wrap(fn, x):
    return fn(x)


def test_trace_fn_with_arg_to_wrap():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.tensor([1.0])
    graph = trace_graph(_fn_with_arg_to_wrap, args=(torch.sin, x))
    original_ret = ivy.to_numpy(_fn_with_arg_to_wrap(torch.sin, x))
    traced_ret = ivy.to_numpy(graph(torch.sin, x))
    assert np.allclose(original_ret, traced_ret)


def fn_with_tracked_slices_inp(x):
    indices = [[1]]
    indices = tf.constant(indices)
    out_shape = x.shape[indices.shape[-1] :]
    out = tf.random.uniform(shape=out_shape)
    return out


def test_trace_fn_with_tracked_slices_inp():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    # raw values
    nc_x = tf.random.uniform(shape=(2, 3, 3))
    c_x = tf.random.uniform(shape=(2, 2, 2))
    # trace
    fname = "test_trace_fn_with_tracked_slices_inp"
    ## int
    graph = trace_graph(
        fn_with_tracked_slices_inp,
        args=(nc_x,),
    )
    # value test
    nc_ret = fn_with_tracked_slices_inp(c_x)
    c_ret = graph(c_x)
    assert nc_ret.shape == c_ret.shape


def inner_fn_with_nested_list_output(x, y):
    x_i, y_i = jax.numpy.meshgrid(x, y, copy=True, sparse=False, indexing="ij")
    z = jax.numpy.stack([x_i, y_i], axis=-1, out=None)
    return z


def test_trace_inner_fn_with_nested_list_output():
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    # raw values
    nc_x1, nc_x2 = (jax.numpy.array([1, 2, 3, 4, 5]), jax.numpy.array([6, 7, 8, 9, 10]))
    c_x1, c_x2 = (
        jax.numpy.array([10, 20, 30, 40, 50]),
        jax.numpy.array([60, 70, 80, 90, 100]),
    )
    # trace
    fname = "test_trace_inner_fn_with_nested_list_output"
    ## int
    graph = trace_graph(
        inner_fn_with_nested_list_output,
        args=(nc_x1, nc_x2),
    )
    # value test
    nc_ret = inner_fn_with_nested_list_output(c_x1, c_x2)
    c_ret = graph(c_x1, c_x2)
    assert np.allclose(nc_ret, c_ret)
    assert np.allclose(nc_ret.shape, c_ret.shape)


def fn_with_cached_tracked_var(x):
    z = 0 + y.shape[0]
    return x[:z]


y = tf.constant([2.0])


@pytest.mark.parametrize("array_caching", ["True", "False"])
def test_trace_fn_with_cached_tracked_var(array_caching):
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    # raw values
    nc_x = tf.random.uniform(shape=(2,), dtype=tf.float32)
    c_x = tf.random.uniform(shape=(3,), dtype=tf.float32)
    # trace
    fname = "test_trace_fn_with_cached_tracked_var"
    ## int
    graph = trace_graph(
        fn_with_cached_tracked_var,
        args=(nc_x,),
        array_caching=array_caching,
    )
    # value test
    nc_ret = fn_with_cached_tracked_var(c_x)
    c_ret = graph(c_x)
    assert np.allclose(nc_ret, c_ret)
    assert np.allclose(nc_ret.shape, c_ret.shape)


def nonzero_tuple(x):
    return x.nonzero(as_tuple=True)


def test_one_item_tuple_return():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    # the tuple should not be removed from the output of these fns
    x = torch.rand(3)
    func_out = nonzero_tuple(x)

    graph = trace_graph(nonzero_tuple, args=((x,)))
    graph_out = graph(x)

    assert type(graph_out) == type(func_out)
    assert torch.all(torch.eq(graph_out[0], func_out[0]))


RETNAMEDTUPLE = namedtuple("OutNamedTuple", ["x", "y"])


def fn_with_named_tuple_in_output(w, x):
    y = w * x
    return RETNAMEDTUPLE(x=x, y=y)


@pytest.mark.parametrize("array_caching", ["True", "False"])
def test_named_tuples_in_output(array_caching):
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    ivy.set_backend("torch")

    (
        nc_x,
        nc_w,
    ) = torch.randn(
        (2, 2)
    ), torch.ones((2, 2))
    c_x, c_w = torch.randn((3, 3)), torch.ones((3, 3))
    graph = trace_graph(
        fn_with_named_tuple_in_output, args=(nc_x, nc_w), array_caching=array_caching
    )
    ret = fn_with_named_tuple_in_output(c_x, c_w)
    comp_ret = graph(c_x, c_w)

    assert np.allclose(ret.x, comp_ret.x)
    assert np.allclose(ret.y, comp_ret.y)
    assert isinstance(comp_ret, tuple) and hasattr(comp_ret, "_fields")


def _iterable_output_with_untracked(x, y, z, w, u):
    # a will be tracked tensor, b will be untracked `None`
    a, b = torch.nn.functional.multi_head_attention_forward(
        x, x, x, 16, 1, y, z, None, None, False, 0.0, w, u, need_weights=False
    )
    return a + a


def test_iterable_output_with_untracked():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = torch.rand((8, 2, 16))
    y = torch.rand((48, 16))
    z = torch.rand((48,))
    w = torch.rand((16, 16))
    u = torch.rand((16,))

    graph = trace_graph(_iterable_output_with_untracked, args=(x, y, z, w, u))

    non_traced_ret = ivy.to_numpy(_iterable_output_with_untracked(x, y, z, w, u))
    traced_ret = ivy.to_numpy(graph(x, y, z, w, u))
    assert np.allclose(non_traced_ret, traced_ret)


def _ivy_linalg(x):
    return ivy.linalg.det(x)


def test_trace_ivy_linalg():
    x = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    graph = trace_graph(_ivy_linalg, args=(x,), to="ivy")
    original_ret = ivy.to_numpy(_ivy_linalg(x))
    traced_ret = ivy.to_numpy(graph(x))
    assert np.allclose(original_ret, traced_ret)


def reduce_dims_numpy(x):
    return np.add.reduce(x, 0)


def test_trace_numpy_ufunc():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()

    x = np.random.randn(5, 10, 15)
    result = reduce_dims_numpy(x)

    graph = trace_graph(reduce_dims_numpy, args=(x,))
    comp_result = graph(x)
    assert np.allclose(result, comp_result)


def test_compiling_fns_directly():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    graph = trace_graph(torch.zeros, args=((3, 3),))
    orignal_ret = ivy.to_numpy(torch.zeros((2, 2)))
    traced_ret = ivy.to_numpy(graph((2, 2)))
    assert np.allclose(orignal_ret, traced_ret)

    graph = trace_graph(ivy.zeros, args=((3, 3),), to="ivy")
    orignal_ret = ivy.to_numpy(ivy.zeros((2, 2)))
    traced_ret = ivy.to_numpy(graph((2, 2)))
    assert np.allclose(orignal_ret, traced_ret)


def numpy_inplace_fn(x, s):
    x.setflags(write=True)
    x.resize((s), refcheck=False)
    x.partition(0)
    x.fill(s)
    x.itemset(0, 2)
    x.sort()
    return x


def test_compiling_numpy_w_inplace():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()

    x = np.array([[0, 1], [1, 0]])
    s = 5
    result = numpy_inplace_fn(x.copy(), s)

    graph = trace_graph(numpy_inplace_fn, args=(x.copy(), s))
    comp_result = graph(x.copy(), s)
    graph_fns = [f.__name__ for f in graph._functions]

    assert np.allclose(result, comp_result)
    assert "setflags" in graph_fns
    assert "resize" in graph_fns
    assert "partition" in graph_fns
    assert "fill" in graph_fns
    assert "itemset" in graph_fns
    assert "sort" in graph_fns


def _fn_to_convert(x, y):
    a = ivy.add(x, y)
    b = x * x
    c = ivy.matmul(a, b)
    return c


@pytest.mark.parametrize("to_ivy", [True, False])
def test_graph_exchange_and_pickling(to_ivy):
    if ivy.current_backend_str() == "paddle":
        # TODO: graph exchange doesn't work for paddle
        pytest.skip()
    x = ivy.random_uniform(shape=(3, 3))
    to = "ivy" if to_ivy else ivy.current_backend_str()
    original_graph = trace_graph(_fn_to_convert, args=(x, x), to=to)

    graph_dict = _convert_graph_to_dict(original_graph)
    pickle.dumps(graph_dict)
    new_graph = _convert_dict_to_graph(graph_dict)
    new_graph.list_function_frequencies()

    x = ivy.random_uniform(shape=(3, 3))
    original_ret = ivy.to_numpy(_fn_to_convert(x, x))
    exchanged_ret = ivy.to_numpy(new_graph(x, x))
    assert np.allclose(original_ret, exchanged_ret)


def _test_jit_split(x):
    return ivy.split(x, num_or_size_splits=[1, 1], axis=2)


# ToDo: not really a tracer test, remove when jitting added to ivy tests
# or transpiling transformers added to tracer ci
def test_jit_split():
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    x = ivy.native_array([[[1, 2, 3]]])
    graph = trace_graph(_test_jit_split, args=(x,))
    graph(x)


def test_trace_errors():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    ivy.unset_backend()
    message = re.escape(
        "The source framework must be specified either with the 'to' argument, "
        "which can be one from ['torch', 'tensorflow', 'jax', 'numpy', 'paddle', 'ivy'], "
        "or by setting ivy's backend with ivy.set_backend('torch'), for example."
    )
    with pytest.raises(ivy.exceptions.IvyException, match=message):
        trace_graph(_functional_fn, args=(jnp.array([1, 2, 3]),))

    lazy_graph = trace_graph(_functional_fn)
    with pytest.raises(ivy.exceptions.IvyException, match=message):
        lazy_graph(jnp.array([1, 2, 3]))

    message = re.escape(
        "'to' must be one of 'torch', 'tensorflow', 'jax', 'numpy', 'paddle' or 'ivy'. "
    )
    with pytest.raises(ivy.exceptions.IvyException, match=message):
        trace_graph(
            _functional_fn, args=(jnp.array([1, 2, 3]),), to="incorrect_backend"
        )


def _fn_returning_constant(x):
    constant = torch.tensor([1.0, 2.0])
    return x + constant, constant


def test_trace_fn_returning_constant():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = torch.tensor([1.0, 2.0])
    graph = trace_graph(_fn_returning_constant, args=(x,))
    graph(x)


def test_pickling_graph():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    ivy.previous_backend()
    model = torch.nn.Linear(2, 3)
    torch_random = torch.randn(1, 2)
    comp_model = trace_graph(model, to="torch", args=(torch_random,))

    ser_dump = dill.dumps(comp_model)
    from ivy.tracer.wrapping import FUNC_TO_PATH

    FUNC_TO_PATH.clear()
    graph = dill.loads(ser_dump)
    graph(torch_random)
    dump = dill.dumps(graph)
    graph = dill.loads(dump)
    graph(torch_random)


if torch.__version__ >= "2.0.0":

    @torch.compile
    def _torch_compiled(x):
        return torch.add(x, x)

else:

    def _torch_compiled(x):
        return torch.add(x, x)


def test_torch_compile_decorator():
    pytest.skip()  # skipping due to torch.compile problem on actions
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = torch.tensor([1.0, 2.0])
    graph = trace_graph(_torch_compiled, args=(x,), graph_caching=False)
    y = torch.tensor([3.0, 4.0])
    original_ret = _torch_compiled(y)
    traced_ret = graph(y)
    assert np.allclose(ivy.to_numpy(original_ret), ivy.to_numpy(traced_ret))


def _tf_fn(x):
    return tf.add(x, x)


@pytest.mark.parametrize("backend_compile", [True, False])
def test_native_compilation_after_serialization(backend_compile):
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    x = tf.constant([1.0])
    comp_model = trace_graph(_tf_fn, args=(x,), backend_compile=backend_compile)
    if backend_compile:
        assert isinstance(comp_model.graph_call, tf.types.experimental.GenericFunction)

    ser_dump = dill.dumps(comp_model)
    from ivy.tracer.wrapping import FUNC_TO_PATH

    FUNC_TO_PATH.clear()
    graph = dill.loads(ser_dump)
    graph(x)
    if backend_compile:
        print(isinstance(graph.graph_call, tf.types.experimental.GenericFunction))

    dump = dill.dumps(graph)
    graph = dill.loads(dump)
    graph(x)
    if backend_compile:
        print(isinstance(graph.graph_call, tf.types.experimental.GenericFunction))


def _torch_size_fn(x):
    return x.size()


def test_trace_tvp_in_output():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.randn((1, 4, 3, 3))
    graph = trace_graph(_torch_size_fn, args=(x,))

    y = torch.randn((5, 10, 3, 3))
    orig = _torch_size_fn(y)
    comp = graph(y)
    assert orig == comp


def _fn_with_casted_ivy_shape(array, add):
    target_shape = list(array.shape)
    target_shape += add
    return target_shape


@pytest.mark.parametrize("fw", ["torch", "numpy"])
def test_trace_casted_ivy_shape(fw):
    if fw != ivy.current_backend_str():
        pytest.skip()

    ivy.set_backend(fw)

    x = ivy.random_normal(shape=(1, 3, 3))
    y = [4, 5]
    args = (x, y)
    graph = trace_graph(_fn_with_casted_ivy_shape, args=args)

    x = ivy.random_normal(shape=(2, 3, 3))
    y = [4, 5, 6, 7]
    orig = _fn_with_casted_ivy_shape(x, y)
    comp = graph(x, y)
    assert orig == comp


def tolist_fn(array):
    return array.to_list()


def test_trace_tolist():
    if ivy.current_backend_str() == "paddle":
        # todo: currently fails for paddle backend
        pytest.skip()

    x = ivy.random_normal(shape=(1, 3, 3))

    graph = trace_graph(tolist_fn, args=(x,))

    x = ivy.random_normal(shape=(2, 3, 3))
    orig = tolist_fn(x)
    comp = graph(x)

    assert orig == comp


def test_trace_torch_lstm():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    model = torch.nn.LSTM(10, 20, 2)
    x = torch.randn(5, 3, 10)
    graph = trace_graph(model, to="torch", args=(x,))
    x = torch.randn(10, 3, 10)
    nc_ret = model(x)
    c_ret = graph(x)
    assert torch.allclose(nc_ret[0], c_ret[0], atol=1e-4)
    assert torch.allclose(nc_ret[1][0], c_ret[1][0], atol=1e-4)
    assert torch.allclose(nc_ret[1][1], c_ret[1][1], atol=1e-4)


def test_avoid_overwrite_to_call():
    comp_model = trace_graph(ivy.add, args=(1, 2))
    # __call__ is only a method wrapper without presence in dict unless
    # an overwrite occurs
    assert "__call__" not in comp_model.__dict__


def test_trace_max_list():
    if ivy.current_backend_str() == "paddle":
        # todo: currently fails for paddle backend
        pytest.skip()

    def fn(x):
        return max(x.shape), x

    a = ivy.ones((1, 2))
    traced_fn = trace_graph(fn, args=(a,))
    b = ivy.ones((3, 4))
    assert traced_fn(b)[0] == 4


def test_einops_cache():
    if ivy.current_backend_str() not in ["torch", "numpy"]:
        # todo: einops tracing fails for for jax/tf/paddle
        pytest.skip()
    from einops import rearrange

    def fn(x):
        return rearrange(x, "a -> a")

    x = ivy.to_native(ivy.ones((1,)))
    graph = trace_graph(fn, args=(x,))
    y = ivy.to_native(ivy.ones((2,)))
    assert graph(y).shape[0] == 2


def fn_w_init(x):
    a = tf.Variable(x.numpy())
    a.assign_add(tf.ones(shape=x.shape))
    return a


# TODO: get init tracing working for tf.Variable/ResourceVariable
# def test_trace_init():
#     if ivy.current_backend_str() != "tensorflow":
#         pytest.skip()

#     x = tf.random.uniform(shape=(2, 2))
#     graph = trace_graph(fn_w_init, args=(x,))
#     x = tf.random.uniform(shape=(4, 4))
#     ret = fn_w_init(x)
#     c_ret = graph(x)
#     assert np.allclose(ret, c_ret)


def fn_w_resource_var(x):
    return x[0] * tf.convert_to_tensor(5.0)


def test_trace_resource_var():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()

    glob.use_reloader = True
    x = tf.Variable(tf.random.uniform(shape=(2, 2)))
    graph = trace_graph(fn_w_resource_var, args=(x,))
    x = tf.Variable(tf.random.uniform(shape=(3, 3)))
    ret = fn_w_resource_var(x)
    c_ret = graph(x)
    assert np.allclose(ret, c_ret)
    glob.use_reloader = False


def _test_compile_torch_methods(x):
    a = x + x
    b = a * x.size()[0]
    return b


def test_compile_torch_methods():
    pytest.skip()  # skipping due to torch.compile problem on actions

    # to ensure our generated source is compatible with torch.compile
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = torch.tensor([1.0, 2.0])
    graph = trace_graph(_test_compile_torch_methods, args=(x,), graph_caching=False)
    opt_graph = torch.compile(graph._scripted_call, fullgraph=True)
    opt_graph(x)


def fn_w_nestable_reprs(x):
    split = ivy.split(x, num_or_size_splits=x.shape[0])
    return split[2]


def test_trace_w_nestable_reprs():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    x = ivy.random.random_uniform(low=0, high=2, shape=(5, 4))
    graph = trace_graph(fn_w_nestable_reprs, args=(x,))
    x = ivy.random.random_uniform(low=0, high=2, shape=(10, 4))
    ret = fn_w_nestable_reprs(x)
    c_ret = graph(x)
    assert torch.allclose(ret, c_ret)


def _fn_list_of_tensors(x):
    a = tf.unstack(x, axis=0)  # returns list of tensors
    return tf.concat(a, axis=0)


def test_trace_list_of_tensors():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    g = trace_graph(_fn_list_of_tensors, to="tensorflow", args=(tf.ones((2, 2)),))

    orig = _fn_list_of_tensors(tf.ones((3, 3)))
    traced = g(tf.ones((3, 3)))
    assert tf.experimental.numpy.allclose(orig, traced)

    orig = _fn_list_of_tensors(tf.ones((1, 1)))
    traced = g(tf.ones((1, 1)))
    assert tf.experimental.numpy.allclose(orig, traced)


def _fn_tuple_of_tensors(x):
    return [torch.unbind(x, dim=0)]


def test_trace_tuple_of_tensors():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    graph = trace_graph(_fn_tuple_of_tensors, to="torch", args=(torch.ones((2, 2)),))

    orig = _fn_tuple_of_tensors(torch.ones((1, 1)))
    traced = graph(torch.ones((1, 1)))
    assert torch.allclose(orig[0][0], traced[0][0])

    orig = _fn_tuple_of_tensors(torch.ones((3, 3)))
    traced = graph(torch.ones((3, 3)))
    assert torch.allclose(orig[0][0], traced[0][0])


def _tensor_list_output_fn(x):
    x = tf.unstack(x)
    return x[0] + x[-1]


def test_track_tensor_list_output():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()

    graph = trace_graph(_tensor_list_output_fn, args=(tf.random.uniform((10, 4)),))

    tf_input = tf.random.uniform((15, 6))
    orig = _tensor_list_output_fn(tf_input)
    traced = graph(tf_input)

    assert np.allclose(orig, traced)


def _fn_w_uncachable_branch(x):
    gi = torch.tensor([[1]])[torch.tensor(0)]
    mul = 2 * gi
    sub = torch.subtract(mul, torch.tensor(1), out=gi)
    return sub + x


def test_fn_w_uncachable_branch():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    x = torch.tensor([1])
    graph = trace_graph(_fn_w_uncachable_branch, to="torch", args=(x,))

    x = torch.tensor([2])
    original_ret = _fn_w_uncachable_branch(x)
    traced_ret = graph(x)
    assert torch.allclose(original_ret, traced_ret)


def _fn_with_cached_ellipsis(x):
    c = (Ellipsis, 0, None)
    return x[c]


def test_fn_with_cached_ellipsis():
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    graph = trace_graph(_fn_with_cached_ellipsis, args=(torch.rand((1, 2, 2)),))
    torch.compile(graph)(torch.rand((1, 2, 2)))


def _fn_w_size_instance_check(size):
    if isinstance(size, torch.Size):
        size = torch.tensor(size)
    return size.float()


def test_fn_w_size_instance_check():
    if ivy.current_backend_str() != "torch":
        pytest.skip()

    s = torch.Size([1, 2, 3])
    trace_graph(_fn_w_size_instance_check, to="torch", args=(s,))


def test_inspecting_generated_source():
    x = ivy.array([1.0])
    graph = trace_graph(lambda x: x + x, args=(x,))
    inspect.getsource(graph._scripted_call)


def _ndarray_method_fn(x):
    x = x.astype(np.float64)
    x = np.abs(x)
    return x


def test_ndarray_method_tracing_with_arrays_and_scalars():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()

    trace_args = (np.asarray(np.float32(1.0)),)
    graph = trace_graph(_ndarray_method_fn, args=trace_args)

    # checks the call still runs on a scalar without throwing an exception
    graph(np.float32(1.0))

    assert "astype" in [f.__name__ for f in graph._functions]
