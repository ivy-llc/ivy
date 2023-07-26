import pytest
import inspect
import warnings
import re

from ivy_tests.test_ivy.helpers import globals as test_globals


@pytest.fixture(autouse=True)
def run_around_tests(request, on_device, backend_fw, frontend, compile_graph, implicit):
    try:
        test_globals.setup_frontend_test(
            frontend,
            backend_fw,
            on_device,
            (
                request.function.test_data
                if hasattr(request.function, "test_data")
                else None
            ),
        )
    except Exception as e:
        test_globals.teardown_frontend_test()
        raise RuntimeError(f"Setting up test for {request.function} failed.") from e
    yield
    test_globals.teardown_frontend_test()


_match_parameters_regex = re.compile(r"\(([^)]+)\)")


def _get_frontend_framework_fn_args(fn):
    try:
        args = list(inspect.signature(fn).parameters)
        introspected = True
    except ValueError:
        # Parse the docstring
        docstring = inspect.getdoc(fn)
        match = _match_parameters_regex.search(docstring)
        if match is not None:
            # 1, -1 skip parenthesis
            args = match.group()[1:-1].split(", ")
            args = list(filter(lambda x: x not in ["*", "/"], args))
        introspected = False

    return args, introspected


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    # Check for arguments mis-match
    if hasattr(pyfuncitem._obj, "_is_frontend_test"):
        test_data = pyfuncitem._obj.test_data

        ivy_args = list(inspect.signature(test_data.ivy_function).parameters)
        fw_args, ret_code = _get_frontend_framework_fn_args(test_data.fw_function)

        if ret_code:
            assert len(ivy_args) == len(
                fw_args
            ), f"Number of arguments mis-match. Expected {fw_args}. got {ivy_args}"
        else:
            # If the function could not be introspected and string is parsed
            # We should do soft-checking as docstring is not reliable
            if len(ivy_args) != len(fw_args):
                warnings.warn(
                    f"Number of arguments mis-match. Expected {fw_args}. got {ivy_args}"
                )
    yield
