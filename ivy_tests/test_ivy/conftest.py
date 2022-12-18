# global
import os
import pytest
from typing import Dict

# local
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy import DefaultDevice
from ivy_tests.test_ivy.helpers import globals as test_globals
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks


GENERAL_CONFIG_DICT = {}
UNSET_TEST_CONFIG = {"list": [], "flag": []}
UNSET_TEST_API_CONFIG = {"list": [], "flag": []}

TEST_PARAMS_CONFIG = []

if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


def pytest_configure(config):
    # device
    raw_value = config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = config.getoption("--backend")
    if raw_value == "all":
        backend_strs = available_frameworks
    else:
        backend_strs = raw_value.split(",")

    # compile_graph
    raw_value = config.getoption("--compile_graph")
    if raw_value == "both":
        compile_modes = [True, False]
    elif raw_value == "true":
        compile_modes = [True]
    else:
        compile_modes = [False]

    # implicit
    raw_value = config.getoption("--with_implicit")
    if raw_value == "true":
        implicit_modes = [True, False]
    else:
        implicit_modes = [False]

    # create test configs
    for backend_str in backend_strs:
        for device in devices:
            for compile_graph in compile_modes:
                for implicit in implicit_modes:
                    TEST_PARAMS_CONFIG.append(
                        (
                            device,
                            test_globals.FWS_DICT[backend_str](),
                            compile_graph,
                            implicit,
                        )
                    )

    process_cl_flags(config)


@pytest.fixture(autouse=True)
def run_around_tests(request, on_device, backend_fw, compile_graph, implicit):
    if hasattr(request.function, "test_data"):
        try:
            test_globals.setup_api_test(
                request.function.test_data,
                backend_fw.backend,
                request.function.ground_truth_backend,
            )
        except Exception as e:
            test_globals.teardown_api_test()
            raise RuntimeError(f"Setting up test for {request.function} failed.") from e
        with backend_fw.use:
            with DefaultDevice(on_device):
                yield
        test_globals.teardown_api_test()
    else:
        with backend_fw.use:
            with DefaultDevice(on_device):
                yield


def pytest_generate_tests(metafunc):
    metafunc.parametrize(
        "on_device,backend_fw,compile_graph,implicit", TEST_PARAMS_CONFIG
    )


def process_cl_flags(config) -> Dict[str, bool]:
    getopt = config.getoption
    no_extra_testing = getopt("--no-extra-testing")

    tmp_config = {
        "as_variable": (
            getopt("--skip-variable-testing"),
            getopt("--with-variable-testing"),
        ),
        "native_array": (
            getopt("--skip-native-array-testing"),
            getopt("--with-native-array-testing"),
        ),
        "with_out": (
            getopt("--skip-out-testing"),
            getopt("--with-out-testing"),
        ),
        "container": (
            getopt("--skip-nestable-testing"),
            getopt("--with-nestable-testing"),
        ),
        "instance_method": (
            getopt("--skip-instance-method-testing"),
            getopt("--with-instance-method-testing"),
        ),
        "test_gradients": (
            getopt("--skip-gradient-testing"),
            getopt("--with-gradient-testing"),
        ),
    }

    # final mapping for hypothesis value generation
    for k, v in tmp_config.items():
        # when both flags are true
        if v[0] and v[1]:
            raise Exception(
                f"--skip-{k}--testing and --with-{k}--testing flags cannot be used \
                    together"
            )
        if v[1] and no_extra_testing:
            raise Exception(
                f"--with-{k}--testing and --no-extra-testing flags cannot be used \
                    together"
            )
        # skipping a test
        if v[0] or no_extra_testing:
            pf.build_flag(k, False)
        # extra testing
        if v[1]:
            pf.build_flag(k, True)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("-B", "--backend", action="store", default="all")
    parser.addoption("--compile_graph", action="store_true")
    parser.addoption("--with_implicit", action="store_true")

    parser.addoption("--skip-variable-testing", action="store_true")
    parser.addoption("--skip-native-array-testing", action="store_true")
    parser.addoption("--skip-out-testing", action="store_true")
    parser.addoption("--skip-nestable-testing", action="store_true")
    parser.addoption("--skip-instance-method-testing", action="store_true")
    parser.addoption("--skip-gradient-testing", action="store_true")

    parser.addoption("--with-variable-testing", action="store_true")
    parser.addoption("--with-native-array-testing", action="store_true")
    parser.addoption("--with-out-testing", action="store_true")
    parser.addoption("--with-nestable-testing", action="store_true")
    parser.addoption("--with-instance-method-testing", action="store_true")
    parser.addoption("--with-gradient-testing", action="store_true")

    parser.addoption("--no-extra-testing", action="store_true")
    parser.addoption(
        "--my_test_dump",
        action="store",
        default=None,
        help="Print test items in my custom format",
    )


def pytest_collection_finish(session):
    # Make sure we're not accidentally accessing it during test
    global TEST_PARAMS_CONFIG
    del TEST_PARAMS_CONFIG
    if session.config.option.my_test_dump is not None:
        for item in session.items:
            item_path = os.path.relpath(item.path)
            print("{}::{}".format(item_path, item.name))
        pytest.exit("Done!")
