import pytest

from ivy import DefaultDevice
from ivy_tests.test_ivy.helpers import globals as test_globals


@pytest.fixture(autouse=True)
def run_around_tests(request, on_device, backend_fw, frontend, compile_graph, implicit):
    if hasattr(request.function, "test_data"):
        try:
            test_globals.setup_frontend_test(
                request.function.test_data, frontend, backend_fw.backend, on_device
            )
        except Exception as e:
            test_globals.teardown_frontend_test()
            raise RuntimeError(f"Setting up test for {request.function} failed.") from e
        with backend_fw.use:
            with DefaultDevice(on_device):
                yield
        test_globals.teardown_frontend_test()
    else:
        with backend_fw.use:
            with DefaultDevice(on_device):
                yield
