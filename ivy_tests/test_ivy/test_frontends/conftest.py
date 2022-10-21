import pytest

from ivy import DefaultDevice
from ivy_tests.test_ivy.helpers import globals as test_globals


@pytest.fixture(autouse=True)
def run_around_tests(
    request, device, backend_fw, fixt_frontend_str, compile_graph, fw, implicit
):
    test_globals.set_frontend(fixt_frontend_str)
    # test_globals.set_backend(backend_fw)
    test_globals.set_test_data(request.function.test_data)

    with backend_fw.use:
        with DefaultDevice(device):
            yield

    test_globals.unset_frontend()
    test_globals.unset_backend()
    test_globals.unset_test_data()
