import pytest

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
