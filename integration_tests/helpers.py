import copy
import ivy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import time
import torch

jax.config.update('jax_enable_x64', True)


# Helpers #
# ------- #

def _check_allclose(x, y, tolerance=1e-3):
    """
    Checks that all values are close. Any arrays must already be in numpy format, rather than native framework.
    """

    if type(x) != type(y):
        assert False, f"mistmatched types: {type(x), type(y)}"

    if isinstance(x, np.ndarray):
        assert np.allclose(x, y, atol=tolerance), "numpy array values are not all close"
        return

    if isinstance(x, (list, set, tuple)):
        all([
            _check_allclose(element_x, element_y, tolerance=tolerance) for element_x, element_y in zip(x, y)
        ])
        return

    if isinstance(x, dict):
        keys_same = all([key_x == key_y for key_x, key_y in zip(x.keys(), y.keys())])
        values_same = all([
            _check_allclose(element_x, element_y, tolerance=tolerance)
            for element_x, element_y in zip(x.values(), y.values())
        ])
        assert keys_same and values_same, "keys or values in dict differ"
        return

    if isinstance(x, float):
        assert x - y < tolerance, f"float values differ: {x} != {y}"
        return

    assert x == y, f"values differ: {x} != {y}"


def _native_array_to_numpy(x):
    if isinstance(x, (torch.Tensor, tf.Tensor)):
        return x.numpy()
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return x


def _nest_array_to_numpy(
    nest, shallow=True
):
    return ivy.nested_map(
        lambda x: _native_array_to_numpy(x),
        nest,
        include_derived=True,
        shallow=shallow,
    )


def _array_to_new_backend(
    x,
    target,
):
    """
    Converts a torch tensor to an array/tensor in a different framework.
    If the input is not a torch tensor, the input if returned without modification.
    """

    if isinstance(x, torch.Tensor):
        if target == "torch": return x
        y = x.numpy()
        if target == "jax":
            y = jnp.array(y)
        elif target == "tensorflow":
            y = tf.convert_to_tensor(y)
        return y
    else:
        return x


def _nest_torch_tensor_to_new_framework(
    nest, target, shallow=True
):
    return ivy.nested_map(
        lambda x: _array_to_new_backend(x, target),
        nest,
        include_derived=True,
        shallow=shallow,
    )


def _test_trace_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    backend_compile,
    tolerance=1e-3,
):
    graph = ivy.trace_graph(
        fn,
        to="torch",
        args=trace_args,
        kwargs=trace_kwargs,
        backend_compile=backend_compile,
        graph_caching=True,
    )

    graph_args = copy.deepcopy(test_args)
    graph_kwargs = copy.deepcopy(test_kwargs)

    orig_out = fn(*test_args, **test_kwargs)
    graph_out = graph(*graph_args, **graph_kwargs)

    orig_np = _nest_array_to_numpy(orig_out)
    graph_np = _nest_array_to_numpy(graph_out)

    _check_allclose(orig_np, graph_np, tolerance=tolerance)


def _test_transpile_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
):
    graph = ivy.transpile(
        fn,
        source="torch",
        to=target,
        args=trace_args,
        kwargs=trace_kwargs,
        backend_compile=backend_compile,
        graph_caching=True,
    )

    orig_out = fn(*test_args, **test_kwargs)
    graph_args = _nest_torch_tensor_to_new_framework(test_args, target)
    graph_kwargs = _nest_torch_tensor_to_new_framework(test_kwargs, target)
    graph_out = graph(*graph_args, **graph_kwargs)

    orig_np = _nest_array_to_numpy(orig_out)
    graph_np = _nest_array_to_numpy(graph_out)

    _check_allclose(orig_np, graph_np, tolerance=tolerance)


def _test_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
    mode="transpile"
):
    start_time = time.time()
    if mode == "transpile":
        print(f"\ntesting {fn.__module__}.{fn.__name__} --> {target}")
    if mode == "trace" and target == "torch":
        print(f"\ntesting {fn.__module__}.{fn.__name__} --> traced graph")

    if mode == "trace":
        if target != "torch":
            pytest.skip()

        _test_trace_function(
            fn,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            backend_compile,
            tolerance=tolerance,
        )
    else:
        _test_transpile_function(
            fn,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            target,
            backend_compile,
            tolerance=tolerance,
        )
    time_taken = round(time.time() - start_time, 2)
    print(f"Test Finished in {time_taken} seconds")
