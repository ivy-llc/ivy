import pytest

import ivy


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
