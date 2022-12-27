import sys
import os
import contextlib
import pytest
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks

import ivy


@pytest.mark.parametrize("backend", available_frameworks)
@pytest.mark.parametrize("trace_mode", ["full", "ivy", "frontend"])
@pytest.mark.parametrize("show_func_wrapper", [True, False])
def test_trace_modes(backend, trace_mode, show_func_wrapper):
    filename = "excep_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    ivy.set_backend(backend)
    ivy.set_exception_trace_mode(trace_mode)
    ivy.set_show_func_wrapper_trace_mode(show_func_wrapper)
    x = ivy.array([])
    y = ivy.array([1.0, 3.0, 4.0])
    lines = ""
    with pytest.raises(Exception):
        ivy.functional.frontends.torch.div(x, y)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if trace_mode == "full" and not show_func_wrapper:
        assert "/func_wrapper.py" not in lines
        assert "/ivy/functional/backends" in lines
        if backend not in ["torch", "numpy"]:
            assert "/site-packages" in lines

    if trace_mode == "full" and show_func_wrapper:
        assert "/func_wrapper.py" in lines
        assert "/ivy/functional/backends" in lines
        if backend not in ["torch", "numpy"]:
            assert "/site-packages" in lines

    if (trace_mode == "ivy" or trace_mode == "frontend") and not show_func_wrapper:
        assert "/func_wrapper.py" not in lines
        assert "/site-packages" not in lines

    if (trace_mode == "ivy" or trace_mode == "frontend") and show_func_wrapper:
        if trace_mode == "ivy":
            assert "/func_wrapper.py" in lines
            assert "/site-packages" not in lines
        if trace_mode == "frontend":
            assert "/ivy/functional/backends" not in lines
            assert "/site-packages" not in lines

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize("trace_mode", ["full", "ivy", "frontend"])
def test_set_trace_mode(trace_mode):
    ivy.set_exception_trace_mode(trace_mode)
    ivy.assertions.check_equal(ivy.get_exception_trace_mode(), trace_mode)


@pytest.mark.parametrize("trace_mode", ["full", "ivy", "frontend"])
def test_unset_trace_mode(trace_mode):
    ivy.set_exception_trace_mode(trace_mode)
    ivy.set_exception_trace_mode("ivy")
    ivy.assertions.check_equal(ivy.get_exception_trace_mode(), "ivy")
    ivy.unset_exception_trace_mode()
    ivy.assertions.check_equal(ivy.get_exception_trace_mode(), trace_mode)


@pytest.mark.parametrize("trace_mode", ["full", "ivy", "frontend"])
def test_get_trace_mode(trace_mode):
    ivy.set_exception_trace_mode(trace_mode)
    ivy.set_exception_trace_mode("ivy")
    ivy.assertions.check_equal(ivy.get_exception_trace_mode(), "ivy")
