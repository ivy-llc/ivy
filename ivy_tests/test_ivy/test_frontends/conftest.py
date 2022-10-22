import pytest

from ivy import DefaultDevice
from ivy_tests.test_ivy.helpers import globals as test_globals


@pytest.fixture(autouse=True)
def run_around_tests(
    request, device, backend_fw, fixt_frontend_str, compile_graph, implicit
):
    try:
        test_globals.setup_test(
            request.function.test_data, fixt_frontend_str, backend_fw.backend
        )
    except Exception as e:
        test_globals.teardown_test()
        raise RuntimeError(f"Setting up test for {request.function} failed.") from e
    with backend_fw.use:
        with DefaultDevice(device):
            yield
    test_globals.teardown_test()
