# general
import pytest
import gc
import inspect
from hypothesis import given, settings, strategies as st
from typing import Union


# local
import ivy
from .hypothesis_helpers import number_helpers as nh


frontend_fw = None
cmd_line_args = (
    "with_out",
    "instance_method",
    "test_gradients",
)
cmd_line_args_lists = (
    "as_variable",
    "native_array",
    "container",
)


def get_ivy_numpy():
    """Import Numpy module from ivy"""
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def get_ivy_jax():
    """Import JAX module from ivy"""
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def get_ivy_tensorflow():
    """Import Tensorflow module from ivy"""
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def get_ivy_torch():
    """Import Torch module from ivy"""
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


_ivy_fws_dict = {
    "numpy": lambda: get_ivy_numpy(),
    "jax": lambda: get_ivy_jax(),
    "tensorflow": lambda: get_ivy_tensorflow(),
    "tensorflow_graph": lambda: get_ivy_tensorflow(),
    "torch": lambda: get_ivy_torch(),
}


def get_current_frontend():
    """Returns the current frontend framework, returns None if no frontend is set."""
    return frontend_fw()


@st.composite
def num_positional_args(draw, *, fn_name: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn_name
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args(fn_name="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args(fn_name="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    for i, fn_name_key in enumerate(fn_name.split(".")):
        if i == 0:
            fn = ivy.__dict__[fn_name_key]
        else:
            fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def num_positional_args_from_fn(draw, *, fn: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args_from_fn(fn="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args_from_fn(fn="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    for param in inspect.signature(fn).parameters.values():
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def bool_val_flags(draw, cl_arg: Union[bool, None]):
    if cl_arg is not None:
        return draw(st.booleans().filter(lambda x: x == cl_arg))
    return draw(st.booleans())


def handle_cmd_line_args(test_fn):
    from ivy_tests.test_ivy.conftest import (
        FW_STRS,
        TEST_BACKENDS,
    )

    # first[1:-2] 5 arguments are all fixtures
    @given(data=st.data())
    @settings(max_examples=1)
    def new_fn(data, get_command_line_flags, device, f, fw, *args, **kwargs):
        gc.collect()
        flag, backend_string = (False, "")
        # skip test if device is gpu and backend is numpy
        if "gpu" in device and ivy.current_backend_str() == "numpy":
            # Numpy does not support GPU
            pytest.skip()
        if not f:
            # randomly draw a backend if not set
            backend_string = data.draw(st.sampled_from(FW_STRS))
            f = TEST_BACKENDS[backend_string]()
        else:
            # use the one which is parametrized
            flag = True

        global frontend_fw
        # Reset the global variable,
        # only set if frontend fw is inferred
        frontend_fw = None
        full_fn_test_path = test_fn.__module__.split(".")
        if len(full_fn_test_path) > 2:
            if full_fn_test_path[2] == "test_frontends":
                frontend_fw = TEST_BACKENDS[full_fn_test_path[3][5:]]

        # set backend using the context manager
        with f.use:
            # inspecting for keyword arguments in test function
            for param in inspect.signature(test_fn).parameters.values():
                if param.name in cmd_line_args:
                    kwargs[param.name] = data.draw(
                        bool_val_flags(get_command_line_flags[param.name])
                    )
                elif param.name in cmd_line_args_lists:
                    kwargs[param.name] = [
                        data.draw(bool_val_flags(get_command_line_flags[param.name]))
                    ]
                elif param.name == "fw":
                    kwargs["fw"] = fw if flag else backend_string
                elif param.name == "device":
                    kwargs["device"] = device
            return test_fn(*args, **kwargs)

    return new_fn


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))
